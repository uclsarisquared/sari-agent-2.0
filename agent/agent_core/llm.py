"""OpenAI-compatible endpoint configuration and shared chat-call utilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from dataclasses import dataclass, field
from io import BytesIO
import json
import os
from pathlib import Path
import time
from typing import Literal, Optional
from urllib.parse import urlsplit

from dotenv import load_dotenv
from loguru import logger
from openai import APIConnectionError, APIStatusError, OpenAI, RateLimitError
from PIL import Image
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout

from agent_core.contracts import JSON_BLOCK_PATTERN
from agent_core.models import agent_model


load_dotenv(Path(__file__).resolve().parent.parent.parent / "config.env")


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
    endpoint, key = endpoint_creds()
    if not endpoint:
        raise RuntimeError(
            "OPENAI_API_URL not found (looked in repo-root config.env, then "
            "sari_env_old conda state)"
        )
    return LLMConfig(
        model_id=agent_model(),
        temperature=temperature,
        mode=mode,
        api_key=key,
        base_url=f"{endpoint}/v1",
        max_tokens=1536,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )


def encode_image(image: Image.Image) -> dict:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


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


API_MAX_ATTEMPTS = 10
API_RETRY_DELAYS = (1, 2, 4, 8, 15, 30, 30, 30, 30)
API_RETRY_EXHAUSTED_PATH_ENV = "SARI_API_RETRY_EXHAUSTED_PATH"


def _signal_api_retry_exhaustion(error: Exception) -> None:
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
        "attempts": API_MAX_ATTEMPTS,
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


def call_with_api_retries(operation):
    """Retry transient endpoint failures and re-raise the final original error."""
    for attempt in range(API_MAX_ATTEMPTS):
        try:
            result = operation()
            if attempt:
                logger.info(f"[api-retry] attempt {attempt + 1}/{API_MAX_ATTEMPTS} succeeded")
            return result
        except Exception as error:
            remaining = API_MAX_ATTEMPTS - (attempt + 1)
            if not is_transient_api_error(error) or attempt + 1 == API_MAX_ATTEMPTS:
                if is_transient_api_error(error):
                    _signal_api_retry_exhaustion(error)
                logger.error(
                    f"[api-retry] attempt {attempt + 1}/{API_MAX_ATTEMPTS} failed "
                    f"({type(error).__name__}: {error}); giving up, {remaining} tries left"
                )
                raise
            delay = API_RETRY_DELAYS[attempt]
            logger.warning(
                f"[api-retry] attempt {attempt + 1}/{API_MAX_ATTEMPTS} failed "
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

    def _api_call_with_retry(self, client: OpenAI, messages: list) -> str:
        def request():
            response = client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                extra_body=self.config.extra_body,
            )
            return response.choices[0].message.content

        return call_with_api_retries(request)
