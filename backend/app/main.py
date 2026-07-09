"""Main FastAPI application for Numerical Analysis Platform."""
import asyncio
import logging
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import (
    root_finding,
    nonlinear_system,
    linear_system,
    integration,
    matrix_inverse,
    matrix_decomposition,
    eigenvalues,
)

# Suppress asyncio ConnectionResetError noise from ngrok health-check probes on Windows
if sys.platform == "win32":
    _original_handler = asyncio.get_event_loop().get_exception_handler()

    def _quiet_handler(loop, context):
        exc = context.get("exception")
        msg = context.get("message", "")
        if isinstance(exc, ConnectionResetError) and "forcibly closed" in str(exc):
            return
        if isinstance(msg, str) and "connection_lost" in msg:
            return
        _original_handler(loop, context)

    asyncio.get_event_loop().set_exception_handler(_quiet_handler)

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(
    title="Numerical Analysis Platform",
    description="A comprehensive numerical analysis web platform for learning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(root_finding.router, prefix="/root-finding", tags=["Root Finding"])
app.include_router(nonlinear_system.router, prefix="/nonlinear-system", tags=["Nonlinear System"])
app.include_router(linear_system.router, prefix="/linear-system", tags=["Linear System"])
app.include_router(integration.router, prefix="/integration", tags=["Numerical Integration"])
app.include_router(matrix_inverse.router, prefix="/matrix-inverse", tags=["Matrix Inverse"])
app.include_router(matrix_decomposition.router, prefix="/matrix-decomposition", tags=["Matrix Decomposition"])
app.include_router(eigenvalues.router, prefix="/eigenvalues", tags=["Eigenvalues"])


@app.get("/")
async def root():
    return {"message": "Numerical Analysis Platform API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}