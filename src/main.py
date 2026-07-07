"""FastAPI application and external-service lifecycle."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import AuthenticationError, RateLimitError

from src.api.agent_routes import router as agent_router
from src.api.routes import router
from src.core.config import settings
from src.core.integration_errors import groq_rate_limit_detail
from src.core.logger import configure_logging, logger
from src.db.artifact_store import initialize_artifact_store
from src.db.checkpoint_store import initialize_checkpoint_store
from src.db.dataset_store import initialize_dataset_store
from src.db.evidence_store import initialize_evidence_store
from src.db.mongo_client import close_mongodb, initialize_mongodb, mongodb_ready
from src.db.research_job_store import initialize_research_job_store
from src.orchestration.job_runner import research_job_runner
from src.rag.retriever_setup import close_qdrant, initialize_qdrant, qdrant_ready


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    try:
        settings.validate_runtime()
    except RuntimeError as exc:
        logger.error("External AI configuration incomplete: %s", exc)
    await asyncio.gather(initialize_mongodb(), initialize_qdrant())
    await asyncio.gather(
        initialize_evidence_store(),
        initialize_artifact_store(),
        initialize_dataset_store(),
        initialize_checkpoint_store(),
        initialize_research_job_store(),
    )
    research_job_runner.start()
    logger.info("Application dependencies initialized")
    yield
    await research_job_runner.stop()
    await close_qdrant()
    close_mongodb()


app = FastAPI(
    title=settings.app_name,
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=[
        "Content-Type",
        "Idempotency-Key",
        "Last-Event-ID",
        "X-Description",
        "X-Session-ID",
    ],
)
app.include_router(router)
app.include_router(agent_router)


@app.exception_handler(AuthenticationError)
async def groq_authentication_error(request, exc):
    logging.getLogger(__name__).warning("Groq rejected the configured API key")
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Groq rejected GROQ_API_KEY. Update .env with a valid key and restart the API."
        },
    )


@app.exception_handler(RateLimitError)
async def groq_rate_limit_error(request, exc):
    return JSONResponse(
        status_code=503,
        content={"detail": groq_rate_limit_detail(exc)},
    )


@app.get("/")
async def root():
    return {"message": "Adaptive RAG API is running", "version": "2.0.0"}


@app.get("/health/live")
async def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness(response: Response):
    mongo, qdrant = await asyncio.gather(mongodb_ready(), qdrant_ready())
    ready = mongo and qdrant and settings.groq_configured and settings.tavily_configured
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if ready else "not_ready",
        "mongodb": mongo,
        "qdrant": qdrant,
        "groq_configured": settings.groq_configured,
        "tavily_configured": settings.tavily_configured,
    }
