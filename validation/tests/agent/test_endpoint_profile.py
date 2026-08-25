from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_core import llm


def test_vllm_profile_preserves_legacy_host_port(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("OPENAI_API_URL", "endpoint.example:9123")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "Qwen/test")
    profile = llm.EndpointProfile.from_env()
    assert profile.base_url == "http://endpoint.example:9123/v1"
    assert profile.model == "Qwen/test"
    assert profile.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_vertex_profile_builds_exact_openapi_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "vertex")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    token = object()
    monkeypatch.setattr(llm, "_adc_bearer_token", token)
    profile = llm.EndpointProfile.from_env()
    assert profile.base_url == (
        "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/"
        "endpoints/openapi"
    )
    assert profile.model == "google/gemini-3.1-flash-lite"
    assert profile.api_key is token
    assert profile.extra_body["google"]["thinking_config"]["thinking_level"] == "MINIMAL"


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("google/gemini-3.1-flash-lite", "MINIMAL"),
        ("google/gemini-3.5-flash", "MINIMAL"),
        ("gemini-3.6-flash", "MINIMAL"),
        ("google/gemini-3.7-flash", "LOW"),
        ("gemini-3.10-flash-preview", "LOW"),
        ("gemini-3.7-flash-video-understanding-eap", "LOW"),
        ("gemini-flash-latest", "LOW"),
        ("gemini-3.5-flash-lite", "MINIMAL"),
        ("publishers/google/models/gemini-3.7-flash", "LOW"),
        ("google/gemini-3.1-pro-preview", "LOW"),
        ("gemini-3-pro-preview", "LOW"),
        ("gemini-3-pro-image", "HIGH"),
        ("publishers/google/models/gemini-3.1-pro-preview", "LOW"),
    ],
)
def test_vertex_thinking_level_follows_model(monkeypatch, model, expected):
    monkeypatch.delenv("VERTEX_THINKING_LEVEL", raising=False)
    assert llm.vertex_thinking_level(model) == expected


def test_vertex_thinking_level_env_overrides(monkeypatch):
    monkeypatch.setenv("VERTEX_THINKING_LEVEL", "high")
    assert llm.vertex_thinking_level("google/gemini-3.1-flash-lite") == "HIGH"


@pytest.mark.parametrize("provider", ["unknown", "gemini"])
def test_unknown_provider_is_actionable(monkeypatch, provider):
    monkeypatch.setenv("LLM_PROVIDER", provider)
    with pytest.raises(llm.EndpointConfigurationError, match="vllm.*vertex"):
        llm.EndpointProfile.from_env()


def test_both_providers_share_image_url_shape():
    assert llm.image_url_part(b"png", "image/png") == {
        "type": "image_url", "image_url": {"url": "data:image/png;base64,cG5n"}
    }


class _Credentials:
    def __init__(self):
        self.token = None
        self.expiry = None
        self.refreshes = 0

    def refresh(self, _request):
        self.refreshes += 1
        self.token = f"token-{self.refreshes}"
        self.expiry = datetime.now(timezone.utc) + timedelta(hours=1)


def test_adc_token_refreshes_reuses_and_locks():
    credentials = _Credentials()
    provider = llm.ADCBearerToken(credentials, object())
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert set(pool.map(lambda _: provider(), range(20))) == {"token-1"}
    assert credentials.refreshes == 1
    assert provider() == "token-1"
    credentials.expiry = datetime.now(timezone.utc) - timedelta(seconds=1)
    assert provider() == "token-2"
    assert credentials.refreshes == 2


def test_adc_refresh_failure_is_actionable():
    class BrokenCredentials:
        token = None
        expiry = None

        def refresh(self, _request):
            raise OSError("metadata server unavailable")

    provider = llm.ADCBearerToken(BrokenCredentials(), object())
    with pytest.raises(llm.EndpointConfigurationError, match="refresh Vertex"):
        provider()


def test_adc_proactive_refresh_failure_uses_valid_token_during_cooldown(monkeypatch):
    class Credentials:
        token = "still-valid"
        expiry = datetime.now(timezone.utc) + timedelta(seconds=30)
        refreshes = 0

        def refresh(self, _request):
            self.refreshes += 1
            raise TimeoutError("metadata timeout")

    now = [100.0]
    monkeypatch.setattr(llm.time, "monotonic", lambda: now[0])
    credentials = Credentials()
    provider = llm.ADCBearerToken(credentials, object())
    assert provider() == "still-valid"
    assert provider() == "still-valid"
    assert credentials.refreshes == 1
    now[0] += 61
    assert provider() == "still-valid"
    assert credentials.refreshes == 2


def test_adc_refresh_recovers_after_proactive_failure(monkeypatch):
    class Credentials:
        token = "old"
        expiry = datetime.now(timezone.utc) + timedelta(seconds=30)
        refreshes = 0

        def refresh(self, _request):
            self.refreshes += 1
            if self.refreshes == 1:
                raise TimeoutError("metadata timeout")
            self.token = "new"
            self.expiry = datetime.now(timezone.utc) + timedelta(hours=1)

    now = [100.0]
    monkeypatch.setattr(llm.time, "monotonic", lambda: now[0])
    provider = llm.ADCBearerToken(Credentials(), object())
    assert provider() == "old"
    now[0] += 61
    assert provider() == "new"


def test_adc_expired_token_does_not_hide_refresh_failure():
    class Credentials:
        token = "expired"
        expiry = datetime.now(timezone.utc) - timedelta(seconds=1)

        def refresh(self, _request):
            raise TimeoutError("metadata timeout")

    provider = llm.ADCBearerToken(Credentials(), object())
    with pytest.raises(llm.EndpointConfigurationError) as raised:
        provider()
    assert isinstance(raised.value.__cause__, TimeoutError)


def test_adc_refresh_request_has_twenty_second_timeout():
    calls = []

    def request(*args, **kwargs):
        calls.append((args, kwargs))

    llm._BoundedGoogleAuthRequest(request)("url", timeout=999)
    assert calls[0][1]["timeout"] == 20


def test_adc_provider_is_process_wide(monkeypatch):
    created = []
    token = object()
    monkeypatch.setattr(llm, "_adc_bearer_token", None)
    monkeypatch.setattr(llm, "ADCBearerToken", lambda: created.append(token) or token)
    with ThreadPoolExecutor(max_workers=8) as pool:
        providers = list(pool.map(lambda _: llm.shared_adc_bearer_token(), range(20)))
    assert all(provider is token for provider in providers)
    assert created == [token]
