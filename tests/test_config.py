import pytest

from whisper_server.config import Settings, is_loopback
from whisper_server.session import authorized


def test_loopback_binding_may_run_without_authentication():
    assert is_loopback("127.0.0.1")
    Settings(host="127.0.0.1").validated()
    assert authorized(None, None)


def test_external_binding_requires_token_or_explicit_unsafe_flag():
    with pytest.raises(ValueError, match="WHISPER_API_TOKEN"):
        Settings(host="0.0.0.0").validated()
    Settings(host="0.0.0.0", api_token="secret").validated()
    Settings(host="0.0.0.0", unsafe_allow_unauthenticated=True).validated()


def test_deepgram_token_authentication_is_exact():
    assert authorized("Token secret", "secret")
    assert not authorized("Bearer secret", "secret")
    assert not authorized("Token Secret", "secret")
