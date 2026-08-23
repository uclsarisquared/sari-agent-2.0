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
    monkeypatch.setattr(llm, "ADCBearerToken", lambda: token)
    profile = llm.EndpointProfile.from_env()
    assert profile.base_url == (
        "https://aiplatform.googleapis.com/v1/projects/my-project/locations/global/"
        "endpoints/openapi"
    )
    assert profile.model == "google/gemini-3.1-flash-lite"
    assert profile.api_key is token
    assert profile.extra_body["google"]["thinking_config"]["thinking_level"] == "MINIMAL"


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
