FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAGAS_DO_NOT_TRACK=true \
    HOME=/tmp \
    XDG_CACHE_HOME=/models/cache \
    HF_HOME=/models/huggingface \
    TRANSFORMERS_CACHE=/models/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    FASTEMBED_CACHE_DIR=/models/fastembed \
    RERANKER_CACHE_DIR=/models/flashrank

WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY src ./src
RUN mkdir -p /models/fastembed /models/flashrank /models/huggingface /models/cache /tmp/matplotlib \
    && chown -R app:app /app /models /tmp/matplotlib
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)"

CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1} --proxy-headers"]
