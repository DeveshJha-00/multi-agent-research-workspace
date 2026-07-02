"""FastAPI application and external-service lifecycle."""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.agent_routes import router as agent_router
from src.api.routes import router
from src.core.config import settings
from src.core.logger import configure_logging, logger
from src.db.artifact_store import initialize_artifact_store
from src.db.evidence_store import initialize_evidence_store
from src.db.mongo_client import close_mongodb, initialize_mongodb, mongodb_ready
from src.rag.retriever_setup import close_qdrant, initialize_qdrant, qdrant_ready


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings.validate_runtime()
    await asyncio.gather(initialize_mongodb(), initialize_qdrant())
    await asyncio.gather(initialize_evidence_store(), initialize_artifact_store())
    logger.info("Application dependencies initialized")
    yield
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
    allow_headers=["Content-Type", "X-Description", "X-Session-ID"],
)
app.include_router(router)
app.include_router(agent_router)


@app.get("/")
async def root():
    return {"message": "Adaptive RAG API is running", "version": "2.0.0"}


@app.get("/health/live")
async def liveness():
    return {"status": "ok"}


@app.get("/health/ready")
async def readiness(response: Response):
    mongo, qdrant = await asyncio.gather(mongodb_ready(), qdrant_ready())
    ready = mongo and qdrant and bool(settings.openai_api_key) and bool(settings.tavily_api_key)
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if ready else "not_ready",
        "mongodb": mongo,
        "qdrant": qdrant,
        "openai_configured": bool(settings.openai_api_key),
        "tavily_configured": bool(settings.tavily_api_key),
    }
