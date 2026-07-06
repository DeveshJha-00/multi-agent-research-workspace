"""User-facing diagnostics for external service errors."""

from groq import APIError, RateLimitError


def groq_rate_limit_detail(error: RateLimitError) -> str:
    """Explain Groq free-tier throttling without exposing provider response details."""
    message = str(error).lower()
    if "tokens per day" in message or "requests per day" in message:
        return "The Groq free-tier daily allowance was reached. Try again after the limit resets."
    return "Groq's free-tier request limit was reached. Wait briefly and retry."


def groq_error_detail(error: APIError) -> str:
    """Translate the provider's other common failures into actionable demo guidance."""
    status_code = getattr(error, "status_code", None)
    message = str(error).lower()
    if status_code == 413 or "request too large" in message:
        return "The research context exceeded Groq's free-tier token limit. Use a shorter objective or fewer sources."
    if "tool_use_failed" in message:
        return "Groq could not format a structured response. Retry the request."
    return "Groq could not complete the request. Check the API logs for provider details."
