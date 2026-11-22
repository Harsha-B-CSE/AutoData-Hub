import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.db.session import init_db
from backend.core.exceptions import AppException, app_exception_handler
from backend.api.routes_upload import router as upload_router
from backend.api.routes_config import router as config_router
from backend.api.routes_process import router as process_router
from backend.api.routes_jobs import router as jobs_router
from backend.api.routes_export import router as export_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables are created
    init_db()
    yield
    # Shutdown logic if needed

app = FastAPI(
    title="AUTODATA-HUB Core API",
    description="Intelligent Dual-Engine Dataset Preprocessing and Performance Optimization Platform",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RFC-7807 Exception Handlers
app.add_exception_handler(AppException, app_exception_handler)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception at %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "type": "https://autodata-hub.local/errors/internal-server-error",
            "title": "Internal Server Error",
            "status": 500,
            "detail": "An unexpected error occurred.",
            "instance": request.url.path,
        },
        headers={"Content-Type": "application/problem+json"}
    )

# Register Routers
app.include_router(upload_router)
app.include_router(config_router)
app.include_router(process_router)
app.include_router(jobs_router)
app.include_router(export_router)

@app.get("/api/v1/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "worker_cores_allocated": settings.MAX_WORKER_PROCESSES,
        "max_upload_mb": settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
