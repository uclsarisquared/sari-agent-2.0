"""OpenAI-compatible endpoint configuration and shared chat-call utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import copy
import re
from dataclasses import dataclass, field
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Literal, Optional
from urllib.parse import urlsplit

from dotenv import load_dotenv
from loguru import logger
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from PIL import Image
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout
from jsonschema import SchemaError, ValidationError
from jsonschema.validators import validator_for

from agent_core.contracts import JSON_BLOCK_PATTERN
from agent_core.models import agent_model


load_dotenv(Path(__file__).resolve().parent.parent.parent / "secrets.env")


@dataclass
class LLMConfig:
    """Transport and generation settings for an OpenAI-compatible LLM client."""
    model_id: str = "google/gemini-2.5-flash-preview-05-20"
    temperature: float = 0.5
    mode: Literal["base", "lean"] = "base"
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: Optional[int] = None
    extra_body: Optional[dict] = None
    provider: Literal["vllm", "vertex"] = "vllm"


# Compatibility name retained while callers migrate to the transport-neutral spelling.
OpenRouterConfig = LLMConfig

VERTEX_MODEL = "google/gemini-3.1-flash-lite"
VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
VERTEX_REFRESH_TIMEOUT_S = 20
VERTEX_REFRESH_COOLDOWN_S = 60

_FLASH_RE = re.compile(r"gemini-(\d+)(?:\.(\d+))?-flash(?!-lite)")
_PRO_RE = re.compile(r"gemini-(\d+)(?:\.(\d+))?-pro")


def vertex_thinking_level(model: str) -> str:
    """Pick the lowest thinking level the model accepts.

    Full-Flash >= 3.7 (and the ``gemini-flash-latest`` alias) dropped the MINIMAL
    tier and return 400 "Thinking level is unsupported: THINKING_LEVEL_MINIMAL";
    their floor is LOW. Measured on models where MINIMAL exists it returns ~8x
    fewer output tokens than LOW, so it stays the default elsewhere.

    Pro-tier models (measured: gemini-3.1-pro-preview 400s the same way) can't
    disable thinking at all - per Google's model-reference table, every Pro
    variant rejects MINIMAL, and only the Image variant is HIGH-only (LOW/MEDIUM
    both rejected there too); every other Pro variant accepts LOW.

    VERTEX_THINKING_LEVEL overrides for any model this heuristic misjudges.
    """
    override = (os.getenv("VERTEX_THINKING_LEVEL") or "").strip().upper()
    if override:
        return override
    normalized = model.rsplit("/", 1)[-1].lower()
    if normalized == "gemini-flash-latest":
        return "LOW"
    if _PRO_RE.search(normalized):
        return "HIGH" if "image" in normalized else "LOW"
    match = _FLASH_RE.search(normalized)
    if match and (int(match.group(1)), int(match.group(2) or 0)) >= (3, 7):
        return "LOW"
    return "MINIMAL"


class EndpointConfigurationError(RuntimeError):
    """The selected OpenAI-compatible endpoint is not completely configured."""


Workload = Literal["guard", "localization", "reasoning", "annotation"]

_VERTEX_TOKEN_FLOORS: dict[str, dict[Workload, int]] = {
    "LOW": {"guard": 1024, "localization": 1536, "reasoning": 4096, "annotation": 4096},
    "MEDIUM": {"guard": 2048, "localization": 3072, "reasoning": 6144, "annotation": 6144},
    "HIGH": {"guard": 4096, "localization": 4096, "reasoning": 8192, "annotation": 8192},
}
_VERTEX_TOKEN_OVERRIDE = {
    "guard": "VERTEX_MAX_TOKENS_GUARD",
    "localization": "VERTEX_MAX_TOKENS_LOCALIZATION",
    "reasoning": "VERTEX_MAX_TOKENS_REASONING",
    "annotation": "VERTEX_MAX_TOKENS_ANNOTATION",
}


def effective_max_tokens(
    provider: str, requested: int | None, workload: Workload, thinking_level: str | None = None,
) -> int | None:
    """Apply Vertex's thinking-aware completion floor without changing vLLM caps."""
    if workload not in _VERTEX_TOKEN_OVERRIDE:
        raise ValueError(f"unknown completion workload {workload!r}")
    if requested is not None and (
        isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0
    ):
        raise ValueError("requested max_tokens must be a positive integer or None")
    if provider != "vertex":
        return requested
    override_name = _VERTEX_TOKEN_OVERRIDE[workload]
    raw_override = os.getenv(override_name)
    if raw_override is not None:
        original_override = raw_override
        try:
            override = int(raw_override)
        except ValueError as error:
            raise EndpointConfigurationError(
                f"{override_name} must be an exact positive integer, got {raw_override!r}."
            ) from error
        if override <= 0 or str(override) != original_override:
            raise EndpointConfigurationError(
                f"{override_name} must be an exact positive integer, got {raw_override!r}."
            )
        return override
    level = (thinking_level or "MINIMAL").strip().upper()
    if level == "MINIMAL":
        return requested
    try:
        floor = _VERTEX_TOKEN_FLOORS[level][workload]
    except KeyError as error:
        raise EndpointConfigurationError(
            f"Unsupported Vertex thinking level {level!r}; expected MINIMAL, LOW, MEDIUM, or HIGH."
        ) from error
    return max(requested or 0, floor)


class ADCBearerToken:
    """Thread-safe callable bearer token for the OpenAI client's ``api_key`` hook."""

    def __init__(
        self, credentials=None, request=None, *, refresh_margin_s: int = 300,
        refresh_cooldown_s: int = VERTEX_REFRESH_COOLDOWN_S,
    ):
        if credentials is None:
            try:
                import google.auth
                from google.auth.transport.requests import Request
                credentials, _ = google.auth.default(scopes=[VERTEX_SCOPE])
                request = request or _BoundedGoogleAuthRequest(Request())
            except Exception as error:  # noqa: BLE001
                raise EndpointConfigurationError(
                    "Vertex authentication failed. Configure Application Default Credentials "
                    "(for example GOOGLE_APPLICATION_CREDENTIALS or workload identity)."
                ) from error
        self._credentials = credentials
        self._request = request
        self._refresh_margin_s = refresh_margin_s
        self._refresh_cooldown_s = refresh_cooldown_s
        self._refresh_retry_at = 0.0
        self._lock = threading.Lock()

    def _needs_refresh(self) -> bool:
        from datetime import datetime, timedelta, timezone
        if not getattr(self._credentials, "token", None):
            return True
        expiry = getattr(self._credentials, "expiry", None)
        if expiry is None:
            return False
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry <= datetime.now(timezone.utc) + timedelta(seconds=self._refresh_margin_s)

    def _token_is_valid(self) -> bool:
        from datetime import datetime, timezone
        if not getattr(self._credentials, "token", None):
            return False
        expiry = getattr(self._credentials, "expiry", None)
        if expiry is None:
            return True
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry > datetime.now(timezone.utc)

    def __call__(self) -> str:
        if self._needs_refresh():
            with self._lock:
                if self._needs_refresh():
                    if self._token_is_valid() and time.monotonic() < self._refresh_retry_at:
                        return self._credentials.token
                    try:
                        self._credentials.refresh(self._request)
                    except Exception as error:  # noqa: BLE001
                        if self._token_is_valid():
                            self._refresh_retry_at = time.monotonic() + self._refresh_cooldown_s
                            logger.warning(
                                "Vertex token refresh failed; using the still-valid token and "
                                f"retrying refresh after {self._refresh_cooldown_s}s "
                                f"({type(error).__name__}: {error})"
                            )
                            return self._credentials.token
                        raise EndpointConfigurationError(
                            "Could not refresh Vertex Application Default Credentials."
                        ) from error
                    self._refresh_retry_at = 0.0
        token = getattr(self._credentials, "token", None)
        if not token:
            raise EndpointConfigurationError("Vertex credentials refreshed without an access token.")
        return token


class _BoundedGoogleAuthRequest:
    """Force google-auth refresh traffic to use the authentication timeout budget."""

    def __init__(self, request):
        self._request = request

    def __call__(self, *args, **kwargs):
        kwargs["timeout"] = VERTEX_REFRESH_TIMEOUT_S
        return self._request(*args, **kwargs)


_adc_bearer_token: ADCBearerToken | None = None
_adc_bearer_lock = threading.Lock()


def shared_adc_bearer_token() -> ADCBearerToken:
    """Return the one ADC credential/token provider shared by this process."""
    global _adc_bearer_token
    if _adc_bearer_token is None:
        with _adc_bearer_lock:
            if _adc_bearer_token is None:
                _adc_bearer_token = ADCBearerToken()
    return _adc_bearer_token


@dataclass(frozen=True)
class EndpointProfile:
    """Provider-specific transport and request settings behind one Chat Completions API."""

    provider: Literal["vllm", "vertex"]
    base_url: str
    api_key: Any
    model: str
    extra_body: dict[str, Any]

    @classmethod
    def from_env(
        cls, *, model: str | None = None, base_url: str | None = None,
        api_key: str | None = None, provider: str | None = None,
    ) -> "EndpointProfile":
        selected = (
            provider
            or ("vllm" if base_url or api_key else None)
            or os.getenv("LLM_PROVIDER")
            or "vllm"
        ).strip().lower()
        if selected == "vllm":
            endpoint, fallback_key = endpoint_creds()
            if base_url:
                endpoint = normalize_endpoint_root(base_url)
            key = api_key or fallback_key
            if not endpoint:
                raise EndpointConfigurationError(
                    "LLM_PROVIDER=vllm requires OPENAI_API_URL (scheme/host/port)."
                )
            if not key:
                raise EndpointConfigurationError("LLM_PROVIDER=vllm requires OPENAI_API_KEY.")
            return cls(
                provider="vllm", base_url=f"{endpoint}/v1", api_key=key,
                model=model or agent_model(),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        if selected == "vertex":
            if base_url or api_key:
                raise EndpointConfigurationError(
                    "LLM_PROVIDER=vertex constructs its endpoint internally and uses ADC; "
                    "--base-url/--api-key overrides are not supported."
                )
            project = (os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
            if not project:
                raise EndpointConfigurationError(
                    "LLM_PROVIDER=vertex requires GOOGLE_CLOUD_PROJECT."
                )
            location = (os.getenv("GOOGLE_CLOUD_LOCATION") or "global").strip()
            if not location:
                raise EndpointConfigurationError("GOOGLE_CLOUD_LOCATION cannot be empty.")
            host = ("aiplatform.googleapis.com" if location == "global"
                    else f"{location}-aiplatform.googleapis.com")
            url = (f"https://{host}/v1/projects/{project}/locations/{location}"
                   "/endpoints/openapi")
            selected_model = model or os.getenv("OPENAI_MODEL") or VERTEX_MODEL
            return cls(
                provider="vertex", base_url=url, api_key=shared_adc_bearer_token(),
                model=selected_model,
                extra_body={"google": {"thinking_config": {
                    "thinking_level": vertex_thinking_level(selected_model)}}},
            )
        raise EndpointConfigurationError(
            f"Unknown LLM_PROVIDER={selected!r}; expected 'vllm' or 'vertex'."
        )

    def request_options(self) -> dict[str, Any]:
        return {"extra_body": self.extra_body.copy()}

    @property
    def thinking_level(self) -> str | None:
        if self.provider != "vertex":
            return None
        return (
            self.extra_body.get("google", {})
            .get("thinking_config", {})
            .get("thinking_level")
        )


@dataclass(frozen=True)
class CompletionResult:
    """Lossless completion data needed by conversational provider protocols."""

    text: str | None
    assistant_message: dict[str, Any]
    finish_reason: str | None
    usage: dict[str, Any]
    raw_response: Any
    workload: Workload
    requested_max_tokens: int | None
    effective_max_tokens: int | None
    thinking_level: str | None

    def diagnostic(self) -> str:
        return (
            f"workload={self.workload}, requested_max_tokens={self.requested_max_tokens}, "
            f"effective_max_tokens={self.effective_max_tokens}, "
            f"thinking_level={self.thinking_level}, finish_reason={self.finish_reason}, "
            f"usage={self.usage}"
        )


@dataclass(frozen=True)
class StructuredCompletion:
    value: Any
    completion: CompletionResult
    enforcement: Literal["native", "prompt_fallback"]


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {key: item for key, item in value.items() if item is not None}
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return {key: item for key, item in data.items() if item is not None}
    raise TypeError(f"Unsupported completion object: {type(value).__name__}")


def _completion_result(
    response: Any, *, workload: Workload, requested: int | None,
    effective: int | None, thinking_level: str | None,
) -> CompletionResult:
    choice = response.choices[0]
    message = choice.message
    envelope = ChatEndpoint.envelope(response)
    assistant = _model_dump(message)
    assistant.setdefault("role", getattr(message, "role", "assistant"))
    if "content" not in assistant:
        assistant["content"] = getattr(message, "content", None)
    assistant = {key: value for key, value in assistant.items() if value is not None}
    return CompletionResult(
        text=getattr(message, "content", None), assistant_message=assistant,
        finish_reason=getattr(choice, "finish_reason", None),
        usage=envelope.get("usage", {}) or {}, raw_response=response, workload=workload,
        requested_max_tokens=requested, effective_max_tokens=effective,
        thinking_level=thinking_level,
    )


def completion_result_from_response(
    response: Any, *, provider: str, thinking_level: str | None,
    workload: Workload, requested_max_tokens: int | None,
) -> CompletionResult:
    """Normalize a response from legacy direct-client callers with policy diagnostics."""
    effective = effective_max_tokens(
        provider, requested_max_tokens, workload, thinking_level
    )
    return _completion_result(
        response, workload=workload, requested=requested_max_tokens,
        effective=effective, thinking_level=thinking_level,
    )


def image_url_part(data: bytes, mime_type: str = "image/png") -> dict[str, Any]:
    """Build the OpenAI Chat Completions image part used by every provider."""
    encoded = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


class ChatEndpoint:
    """Non-streaming OpenAI-compatible Chat Completions transport."""

    def __init__(self, profile: EndpointProfile | None = None, *, timeout: float = 180.0):
        self.profile = profile or EndpointProfile.from_env()
        self.client = OpenAI(base_url=self.profile.base_url, api_key=self.profile.api_key,
                             timeout=timeout, max_retries=0)

    def create(
        self, *, messages: list[dict[str, Any]], schema: dict | None = None,
        schema_name: str = "response", model: str | None = None,
        temperature: float = 0.0, max_tokens: int | None = None,
        extra_body: dict | None = None, workload: Workload = "reasoning",
    ):
        if schema is not None:
            return self.create_structured(
                messages=messages, schema=schema, schema_name=schema_name, model=model,
                temperature=temperature, max_tokens=max_tokens, extra_body=extra_body,
                workload=workload,
            ).completion.raw_response
        body = dict(self.profile.extra_body)
        if extra_body:
            body.update(extra_body)
        kwargs: dict[str, Any] = {
            "model": model or self.profile.model, "messages": messages,
            "temperature": temperature, "extra_body": body,
        }
        budget = effective_max_tokens(
            self.profile.provider, max_tokens, workload, self.profile.thinking_level
        )
        if budget is not None:
            kwargs["max_tokens"] = budget
        return self.client.chat.completions.create(**kwargs)

    def create_result(
        self, *, messages: list[dict[str, Any]], model: str | None = None,
        temperature: float = 0.0, max_tokens: int | None = None,
        extra_body: dict | None = None, workload: Workload = "reasoning",
    ) -> CompletionResult:
        response = self.create(
            messages=messages, model=model, temperature=temperature, max_tokens=max_tokens,
            extra_body=extra_body, workload=workload,
        )
        effective = effective_max_tokens(
            self.profile.provider, max_tokens, workload, self.profile.thinking_level
        )
        return _completion_result(
            response, workload=workload, requested=max_tokens, effective=effective,
            thinking_level=self.profile.thinking_level,
        )

    def create_structured(
        self, *, messages: list[dict[str, Any]], schema: dict[str, Any],
        schema_name: str = "response", model: str | None = None,
        temperature: float = 0.0, max_tokens: int | None = None,
        extra_body: dict | None = None, workload: Workload = "reasoning",
        call_name: str | None = None, timeout: float | None = None,
    ) -> "StructuredCompletion":
        return structured_chat_completion(
            client=self.client, provider=self.profile.provider,
            thinking_level=self.profile.thinking_level, default_extra_body=self.profile.extra_body,
            messages=messages, schema=schema, schema_name=schema_name,
            model=model or self.profile.model, temperature=temperature, max_tokens=max_tokens,
            extra_body=extra_body, workload=workload, call_name=call_name or schema_name,
            timeout=timeout,
        )

    @staticmethod
    def envelope(response) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json", exclude_none=False)
        if isinstance(response, dict):
            return response
        usage = getattr(response, "usage", None)
        return {"usage": _model_dump(usage) if usage is not None else {}}


def normalize_endpoint_root(raw: str) -> str:
    """Normalize and validate an endpoint root whose port is configuration-owned."""
    endpoint = raw.strip().rstrip("/")
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"
    parsed = urlsplit(endpoint)
    if not parsed.hostname:
        raise RuntimeError(f"OPENAI_API_URL has no host: {endpoint!r}")
    try:
        port = parsed.port
    except ValueError as error:
        raise RuntimeError(f"OPENAI_API_URL has an invalid port: {endpoint!r}") from error
    if port is None:
        raise RuntimeError(
            "OPENAI_API_URL must include the endpoint port "
            "(for example http://host:8000)"
        )
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise RuntimeError(
            "OPENAI_API_URL must contain only scheme, host, and port; the code appends /v1"
        )
    return f"{parsed.scheme}://{parsed.netloc}"


def endpoint_creds() -> tuple[Optional[str], Optional[str]]:
    """Return the normalized endpoint root (including its port) and bearer key.

    ``OPENAI_API_URL`` owns transport location: scheme, host, and port. Callers
    own the OpenAI API prefix and append ``/v1`` themselves.
    """
    endpoint = os.getenv("OPENAI_API_URL")
    key = os.getenv("OPENAI_API_KEY")
    if not (endpoint and key):
        try:
            conda_state = os.getenv("SARI_CONDA_STATE", r"C:/Sari/sari_env_old/conda-meta/state")
            with open(conda_state, encoding="utf-8") as handle:
                values = json.load(handle).get("env_vars", {})
            endpoint = endpoint or values.get("OPENAI_API_URL")
            key = key or values.get("OPENAI_API_KEY")
        except OSError:
            pass
    if endpoint:
        endpoint = normalize_endpoint_root(endpoint)
    return endpoint, key


def agent_vlm_config(temperature: float = 0.5, mode: str = "lean") -> LLMConfig:
    profile = EndpointProfile.from_env()
    return LLMConfig(
        model_id=profile.model,
        temperature=temperature,
        mode=mode,
        api_key=profile.api_key,
        base_url=profile.base_url,
        max_tokens=1536,
        extra_body=profile.extra_body,
        provider=profile.provider,
    )


def encode_image(image: Image.Image) -> dict:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return image_url_part(buffer.getvalue(), "image/png")


def build_content(*parts) -> list:
    content = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, Image.Image):
            content.append(encode_image(part))
        elif isinstance(part, str):
            content.append({"type": "text", "text": part})
    return content


DEFAULT_API_MAX_ATTEMPTS = 10
API_RETRY_DELAYS = (1, 2, 4, 8, 15)
API_RETRY_EXHAUSTED_PATH_ENV = "SARI_API_RETRY_EXHAUSTED_PATH"
_api_max_attempts = DEFAULT_API_MAX_ATTEMPTS


class MalformedContentError(ValueError):
    """A model response arrived successfully but did not satisfy its caller's contract."""

    def __init__(
        self, message: str, *, content: Any = None,
        completion_result: CompletionResult | None = None,
    ) -> None:
        super().__init__(message)
        self.content = content
        self.completion_result = completion_result


def configure_api_retries(max_attempts: int) -> None:
    """Set the per-call attempt budget for this process."""
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("api max attempts must be an integer of at least 1")
    global _api_max_attempts
    _api_max_attempts = max_attempts


def api_max_attempts() -> int:
    """Return the active process-wide per-call attempt budget."""
    return _api_max_attempts


def api_retry_delay(retry_index: int) -> int:
    """Return the fixed wait before retry number ``retry_index`` (zero based)."""
    if retry_index < 0:
        raise ValueError("retry index cannot be negative")
    return API_RETRY_DELAYS[retry_index] if retry_index < len(API_RETRY_DELAYS) else 30


def _signal_api_retry_exhaustion(
    error: Exception, *, call_name: str, failure_kind: str, attempts: int
) -> None:
    """Tell a supervising bench runner that a transient API outage exhausted its budget.

    Standalone runs do not set the signal path and retain their existing behavior. The distributed
    runner sets it to an attempt-local file and watches for that file so it can stop and requeue the
    attempt even when an intermediate agent layer would otherwise turn the exception into a normal
    unsuccessful leg.
    """
    raw_path = os.getenv(API_RETRY_EXHAUSTED_PATH_ENV)
    if not raw_path:
        return

    signal_path = Path(raw_path)
    temporary = signal_path.with_name(
        f".{signal_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    cause = _transient_error_in_chain(error)
    payload = {
        "attempts": attempts,
        "call_name": call_name,
        "failure_kind": failure_kind,
        "error_type": type(error).__name__,
        "error": str(error),
        "cause_type": type(cause).__name__,
        "cause": str(cause),
        "signaled_at": time.time(),
    }
    try:
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, signal_path)
    except OSError as signal_error:
        logger.error(f"[api-retry] could not write exhaustion signal: {signal_error}")


def _exception_chain(error: Exception):
    """Yield an exception and its explicit/implicit causes without looping."""
    current: BaseException | None = error
    seen: set[int] = set()
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _direct_api_failure_kind(error: BaseException) -> str | None:
    if isinstance(error, MalformedContentError):
        return "malformed_content"
    if isinstance(error, (APITimeoutError, RequestsTimeout, TimeoutError)):
        return "timeout"
    if isinstance(error, RateLimitError):
        return "rate_limit"
    if isinstance(error, (APIConnectionError, RequestsConnectionError, ConnectionError)):
        return "connection"
    if isinstance(error, APIStatusError):
        if error.status_code in (408, 409, 429) or error.status_code >= 500:
            return "rate_limit" if error.status_code == 429 else "http_status"
        return None
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and (
        status_code in (408, 409, 429) or status_code >= 500
    ):
        return "rate_limit" if status_code == 429 else "http_status"
    return None


def _transient_error_in_chain(error: Exception) -> BaseException:
    return next(
        (candidate for candidate in _exception_chain(error)
         if _direct_api_failure_kind(candidate) is not None),
        error,
    )


def is_transient_api_error(error: Exception) -> bool:
    return any(_direct_api_failure_kind(candidate) for candidate in _exception_chain(error))


def api_failure_kind(error: Exception) -> str | None:
    """Normalize retryable failures for logs and Bench exhaustion diagnostics."""
    return next(
        (kind for candidate in _exception_chain(error)
         if (kind := _direct_api_failure_kind(candidate)) is not None),
        None,
    )


def call_with_api_retries(
    operation: Callable[[], Any],
    *,
    call_name: str = "model_call",
    validator: Callable[[Any], Any] | None = None,
    signal_malformed_content_exhaustion: bool = True,
):
    """Retry endpoint and malformed-content failures under one configured budget.

    ``validator`` runs before an attempt is considered successful and may return a transformed
    result. It must raise :class:`MalformedContentError` for retryable response-contract failures;
    ordinary exceptions remain programming errors and fail immediately.

    ``signal_malformed_content_exhaustion`` may be disabled by callers that can surface an
    exhausted response-contract failure as a recoverable tool result. Endpoint/transport failures
    still signal the supervising runner so it can requeue an attempt whose API is unavailable.
    """
    max_attempts = api_max_attempts()
    for attempt in range(max_attempts):
        try:
            result = operation()
            if validator is not None:
                result = validator(result)
            if attempt:
                logger.info(
                    f"[api-retry] call={call_name} attempt={attempt + 1}/{max_attempts} "
                    "succeeded"
                )
            return result
        except Exception as error:
            failure_kind = api_failure_kind(error)
            remaining = max_attempts - (attempt + 1)
            if failure_kind is None or attempt + 1 == max_attempts:
                should_signal = failure_kind is not None and (
                    failure_kind != "malformed_content"
                    or signal_malformed_content_exhaustion
                )
                if should_signal:
                    _signal_api_retry_exhaustion(
                        error,
                        call_name=call_name,
                        failure_kind=failure_kind,
                        attempts=max_attempts,
                    )
                logger.error(
                    f"[api-retry] call={call_name} failure_kind={failure_kind or 'non_retryable'} "
                    f"attempt={attempt + 1}/{max_attempts} failed "
                    f"({type(error).__name__}: {error}); giving up, {remaining} tries left"
                )
                raise
            delay = api_retry_delay(attempt)
            logger.warning(
                f"[api-retry] call={call_name} failure_kind={failure_kind} "
                f"attempt={attempt + 1}/{max_attempts} failed "
                f"({type(error).__name__}: {error}); retrying in {delay}s, {remaining} tries left"
            )
            time.sleep(delay)
    raise AssertionError("retry loop exhausted without returning or raising")


def validate_json_schema(schema: dict[str, Any], *, provider: str) -> Any:
    """Validate a caller-owned schema without changing its meaning."""
    if not isinstance(schema, dict):
        raise ValueError("structured-output schema must be a JSON object")
    try:
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
    except SchemaError as error:
        raise ValueError(f"invalid JSON Schema: {error.message}") from error

    if provider == "vertex":
        def resolve(pointer: str) -> Any:
            if not pointer.startswith("#/"):
                return schema if pointer == "#" else None
            current: Any = schema
            for part in pointer[2:].split("/"):
                key = part.replace("~1", "/").replace("~0", "~")
                if not isinstance(current, dict) or key not in current:
                    return None
                current = current[key]
            return current

        def recursive(node: Any, active: frozenset[int] = frozenset()) -> bool:
            if not isinstance(node, (dict, list)):
                return False
            if id(node) in active:
                return True
            active = active | {id(node)}
            if isinstance(node, list):
                return any(recursive(item, active) for item in node)
            if "$recursiveRef" in node or "$dynamicRef" in node:
                return True
            ref = node.get("$ref")
            target = resolve(ref) if isinstance(ref, str) and ref.startswith("#") else None
            if target is not None and recursive(target, active):
                return True
            return any(
                recursive(value, active) for key, value in node.items() if key != "$ref"
            )

        if recursive(schema):
            raise ValueError(
                "Vertex structured output does not support recursive JSON Schemas; "
                "replace recursive $ref/$recursiveRef/$dynamicRef links with a bounded shape."
            )
    return validator_cls(schema)


def _messages_with_schema_prompt(
    messages: list[dict[str, Any]], schema: dict[str, Any]
) -> list[dict[str, Any]]:
    prompted = copy.deepcopy(messages)
    suffix = "\n\nReply with ONLY a JSON value matching this schema:\n" + json.dumps(
        schema, ensure_ascii=False, separators=(",", ":")
    )
    for message in reversed(prompted):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content + suffix
        elif isinstance(content, list):
            content.append({"type": "text", "text": suffix.lstrip()})
        else:
            message["content"] = suffix.lstrip()
        return prompted
    raise ValueError("structured completion requires a user message for schema instructions")


def _parse_structured(
    completion: CompletionResult, validator: Any, *, tolerant: bool, call_name: str,
) -> Any:
    text = completion.text
    if not isinstance(text, str) or not text.strip():
        raise MalformedContentError(
            f"{call_name} returned empty structured content ({completion.diagnostic()})",
            content=text,
        )
    candidates = [text.strip()]
    if tolerant:
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if fenced:
            candidates.append(fenced.group(1).strip())
        for left, right in (("{", "}"), ("[", "]")):
            start, end = text.find(left), text.rfind(right)
            if start >= 0 and end > start:
                candidates.append(text[start:end + 1])
    last_error: Exception | None = None
    for candidate in dict.fromkeys(candidates):
        parsers = (json.loads, __import__("ast").literal_eval) if tolerant else (json.loads,)
        for parser in parsers:
            try:
                value = parser(candidate)
                validator.validate(value)
                return value
            except (json.JSONDecodeError, SyntaxError, ValueError, ValidationError) as error:
                last_error = error
    detail = getattr(last_error, "message", str(last_error))
    raise MalformedContentError(
        f"{call_name} response violated its schema: {detail} ({completion.diagnostic()})",
        content=text,
    ) from last_error


def _schema_related_bad_request(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status != 400:
        return False
    body = getattr(error, "body", None)
    detail = f"{error} {body}".lower()
    return any(term in detail for term in ("json_schema", "response_format", "schema"))


def structured_chat_completion(
    *, client: Any, provider: Literal["vllm", "vertex"], thinking_level: str | None,
    default_extra_body: dict[str, Any] | None, messages: list[dict[str, Any]],
    schema: dict[str, Any], schema_name: str, model: str, temperature: float = 0.0,
    max_tokens: int | None = None, extra_body: dict[str, Any] | None = None,
    workload: Workload = "reasoning", call_name: str = "structured_completion",
    timeout: float | None = None,
) -> StructuredCompletion:
    """Execute provider-aware structured output, including Vertex's one fallback phase."""
    validator = validate_json_schema(schema, provider=provider)
    effective = effective_max_tokens(provider, max_tokens, workload, thinking_level)
    body = dict(default_extra_body or {})
    if extra_body:
        body.update(extra_body)

    def request(request_messages: list[dict[str, Any]], *, native: bool) -> CompletionResult:
        kwargs: dict[str, Any] = {
            "model": model, "messages": request_messages, "temperature": temperature,
            "extra_body": body,
        }
        if effective is not None:
            kwargs["max_tokens"] = effective
        if timeout is not None:
            kwargs["timeout"] = timeout
        if native:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        response = client.chat.completions.create(**kwargs)
        return _completion_result(
            response, workload=workload, requested=max_tokens, effective=effective,
            thinking_level=thinking_level,
        )

    if provider == "vllm":
        prompted = _messages_with_schema_prompt(messages, schema)

        def qwen_attempt() -> StructuredCompletion:
            completion = request(prompted, native=True)
            value = _parse_structured(
                completion, validator, tolerant=True, call_name=call_name
            )
            return StructuredCompletion(value, completion, "native")

        return call_with_api_retries(qwen_attempt, call_name=call_name)

    fallback_reason: str
    try:
        completion = call_with_api_retries(
            lambda: request(messages, native=True), call_name=f"{call_name}.native"
        )
    except Exception as error:
        if not _schema_related_bad_request(error):
            raise
        fallback_reason = f"schema-related HTTP 400: {error}"
    else:
        try:
            value = _parse_structured(
                completion, validator, tolerant=False, call_name=call_name
            )
            return StructuredCompletion(value, completion, "native")
        except MalformedContentError as error:
            fallback_reason = str(error)

    logger.warning(
        f"[structured-output] call={call_name} provider=vertex enforcement=prompt_fallback "
        f"reason={fallback_reason}"
    )
    prompted = _messages_with_schema_prompt(messages, schema)
    fallback_completion = call_with_api_retries(
        lambda: request(prompted, native=False), call_name=f"{call_name}.prompt_fallback"
    )
    value = _parse_structured(
        fallback_completion, validator, tolerant=True, call_name=f"{call_name}.prompt_fallback"
    )
    return StructuredCompletion(value, fallback_completion, "prompt_fallback")


class BaseAgent(ABC):
    """Shared configuration, reply parsing, and retry support for LLM-backed agents."""
    @abstractmethod
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()

    @property
    def extractable_json_structured_output(self):
        return JSON_BLOCK_PATTERN

    def _api_call_with_retry(
        self,
        client: OpenAI,
        messages: list,
        *,
        call_name: str = "model_call",
        validator: Callable[[str], Any] | None = None,
    ) -> str:
        result = BaseAgent._api_call_result_with_retry(
            self,
            client, messages, call_name=call_name, validator=validator
        )
        return result.text

    def _api_call_result_with_retry(
        self,
        client: OpenAI,
        messages: list,
        *,
        call_name: str = "model_call",
        validator: Callable[[str], Any] | None = None,
        workload: Workload = "reasoning",
    ) -> CompletionResult:
        """Completion-result path used when provider message metadata must survive history."""
        thinking_level = (
            (self.config.extra_body or {}).get("google", {})
            .get("thinking_config", {})
            .get("thinking_level")
        )
        effective = effective_max_tokens(
            self.config.provider, self.config.max_tokens, workload, thinking_level
        )

        def request():
            response = client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=effective,
                extra_body=self.config.extra_body,
            )
            return _completion_result(
                response, workload=workload, requested=self.config.max_tokens,
                effective=effective, thinking_level=thinking_level,
            )

        def validate_result(result: CompletionResult) -> CompletionResult:
            try:
                if validator is not None:
                    validator(result.text)
                elif not isinstance(result.text, str) or not result.text.strip():
                    raise MalformedContentError(
                        f"model response was empty ({result.diagnostic()})", content=result.text
                    )
            except MalformedContentError as error:
                error.completion_result = result
                if "workload=" not in str(error):
                    error.args = (f"{error} ({result.diagnostic()})",)
                raise
            return result

        return call_with_api_retries(
            request, call_name=call_name, validator=validate_result
        )
