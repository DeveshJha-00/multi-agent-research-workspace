from src.core.integration_errors import groq_error_detail, groq_rate_limit_detail


class FakeRateLimit:
    def __str__(self):
        return "Limit exceeded for requests per day"


class FakeRequestLimit:
    code = "rate_limit_exceeded"

    def __str__(self):
        return "Rate limit reached for requests"


def test_daily_limit_reports_reset():
    assert "resets" in groq_rate_limit_detail(FakeRateLimit())


def test_request_rate_limit_recommends_retry():
    assert "retry" in groq_rate_limit_detail(FakeRequestLimit())


class FakeOversizedRequest:
    status_code = 413

    def __str__(self):
        return "Request too large"


def test_oversized_context_has_actionable_message():
    assert "shorter objective or fewer sources" in groq_error_detail(FakeOversizedRequest())
