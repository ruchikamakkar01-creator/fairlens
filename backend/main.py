import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.analyze import router as analyze_router

app = FastAPI(title="FairLens AI Backend", version="0.1.0")
logger = logging.getLogger("fairlens.api")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/api")

# Serve static files from the 'dist' directory (built React app)
app.mount("/", StaticFiles(directory="dist", html=True), name="static")


@app.get("/api/health")
def health_check():
    return {"ok": True, "service": "fairlens-fastapi", "status": "healthy"}


@app.get("/api/ready")
def readiness_check():
    return {"ok": True, "service": "fairlens-fastapi", "status": "ready"}


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(_request: Request, exc: RequestValidationError):
    logger.warning("Request validation failed: %s", exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid request payload.",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    logger.exception("Unhandled backend error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "Something went wrong while processing your request.",
        },
    )
