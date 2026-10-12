import pytest


@pytest.fixture(autouse=True)
def dummy_llm_settings(monkeypatch):
    """
    Dummy LLM settings so that importing `mathutrice.llm_client` succeeds in CI,
    and no real `.env` is read. No test may reach a real endpoint: every client
    a test uses is a mock.
    """
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("LLM_BASE_URL", "http://llm.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "dummy-key")
    monkeypatch.setenv("LLM_MODEL", "dummy-model")
