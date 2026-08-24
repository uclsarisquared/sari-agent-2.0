"""OpenAI-compatible endpoint configuration and shared chat-call utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
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


# Compatibility name retained while callers migrate to the transport-neutral spelling.
OpenRouterConfig = LLMConfig

VERTEX_MODEL = "google/gemini-3.1-flash-lite"
VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

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


class ADCBearerToken:
    """Thread-safe callable bearer token for the OpenAI client's ``api_key`` hook."""

    def __init__(self, credentials=None, request=None, *, refresh_margin_s: int = 300):
        if credentials is None:
            try:
                import google.auth
                from google.auth.transport.requests import Request
                credentials, _ = google.auth.default(scopes=[VERTEX_SCOPE])
                request = request or Request()
            except Exception as error:  # noqa: BLE001
                raise EndpointConfigurationError(
                    "Vertex authentication failed. Configure Application Default Credentials "
                    "(for example GOOGLE_APPLICATION_CREDENTIALS or workload identity)."
                ) from error
        self._credentials = credentials
        self._request = request
        self._refresh_margin_s = refresh_margin_s
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

    def __call__(self) -> str:
        if self._needs_refresh():
            with self._lock:
                if self._needs_refresh():
                    try:
                        self._credentials.refresh(self._request)
                    except Exception as error:  # noqa: BLE001
                        raise EndpointConfigurationError(
                            "Could not refresh Vertex Application Default Credentials."
                        ) from error
        token = getattr(self._credentials, "token", None)
        if not token:
            raise EndpointConfigurationError("Vertex credentials refreshed without an access token.")
        return token


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
        selected = (provider or os.getenv("LLM_PROVIDER") or "vllm").strip().lower()
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
                provider="vertex", base_url=url, api_key=ADCBearerToken(),
                model=selected_model,
                extra_body={"google": {"thinking_config": {
                    "thinking_level": vertex_thinking_level(selected_model)}}},
            )
        raise EndpointConfigurationError(
            f"Unknown LLM_PROVIDER={selected!r}; expected 'vllm' or 'vertex'."
        )

    def request_options(self) -> dict[str, Any]:
        return {"extra_body": self.extra_body.copy()}


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
        extra_body: dict | None = None,
    ):
        body = dict(self.profile.extra_body)
        if extra_body:
            body.update(extra_body)
        kwargs: dict[str, Any] = {
            "model": model or self.profile.model, "messages": messages,
            "temperature": temperature, "extra_body": body,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": schema, "strict": True},
            }
        return self.client.chat.completions.create(**kwargs)

    @staticmethod
    def envelope(response) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="json", exclude_none=False)
        if isinstance(response, dict):
            return response
        raise TypeError(f"Unsupported Chat Completions response: {type(response).__name__}")


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

    def __init__(self, message: str, *, content: Any = None) -> None:
        super().__init__(message)
        self.content = content


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
    payload = {
        "attempts": attempts,
        "call_name": call_name,
        "failure_kind": failure_kind,
        "error_type": type(error).__name__,
        "error": str(error),
        "signaled_at": time.time(),
    }
    try:
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, signal_path)
    except OSError as signal_error:
        logger.error(f"[api-retry] could not write exhaustion signal: {signal_error}")


def is_transient_api_error(error: Exception) -> bool:
    if isinstance(
        error,
        (
            APIConnectionError,
            RateLimitError,
            RequestsConnectionError,
            RequestsTimeout,
            TimeoutError,
            ConnectionError,
        ),
    ):
        return True
    if isinstance(error, APIStatusError):
        return error.status_code in (408, 409, 429) or error.status_code >= 500
    # Raw requests callers raise HTTPError, whose status lives on its response rather than the
    # exception. Keep this structural so the shared retry helper is not tied to one HTTP client.
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code in (408, 409, 429) or status_code >= 500
    return False


def api_failure_kind(error: Exception) -> str | None:
    """Normalize retryable failures for logs and Bench exhaustion diagnostics."""
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


def call_with_api_retries(
    operation: Callable[[], Any],
    *,
    call_name: str = "model_call",
    validator: Callable[[Any], Any] | None = None,
):
    """Retry endpoint and malformed-content failures under one configured budget.

    ``validator`` runs before an attempt is considered successful and may return a transformed
    result. It must raise :class:`MalformedContentError` for retryable response-contract failures;
    ordinary exceptions remain programming errors and fail immediately.
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
                if failure_kind is not None:
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
        def request():
            response = client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                extra_body=self.config.extra_body,
            )
            return response.choices[0].message.content

        return call_with_api_retries(request, call_name=call_name, validator=validator)
