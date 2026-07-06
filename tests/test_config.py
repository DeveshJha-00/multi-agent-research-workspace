import pytest

from src.core.config import Settings


def test_comma_separated_cors_origins_are_parsed():
    config = Settings(
        _env_file=None,
        cors_origins="https://one.example, https://two.example",
    )
    assert config.cors_origins == ["https://one.example", "https://two.example"]


def test_runtime_configuration_rejects_missing_keys():
    config = Settings(_env_file=None, groq_api_key="", tavily_api_key="")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY, TAVILY_API_KEY"):
        config.validate_runtime()


def test_runtime_configuration_rejects_example_placeholders():
    config = Settings(
        _env_file=None,
        groq_api_key="replace-me",
        tavily_api_key="your-tavily-key",
    )
    with pytest.raises(RuntimeError, match="GROQ_API_KEY, TAVILY_API_KEY"):
        config.validate_runtime()


def test_overlap_must_be_smaller_than_chunk():
    config = Settings(
        _env_file=None,
        groq_api_key="key",
        tavily_api_key="key",
        chunk_size=500,
        chunk_overlap=500,
    )
    with pytest.raises(RuntimeError, match="CHUNK_OVERLAP"):
        config.validate_runtime()
