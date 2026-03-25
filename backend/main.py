"""
FastAPI application entry point.

Thin entry: creates app, registers routers, adds CORS.
No business logic here (Constitution §V).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import images


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="FT Mixer & Emphasizer",
        description="Fourier Transform Mixer and Properties Emphasizer API",
        version="1.0.0",
    )

    # CORS for Vite dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(images.router, prefix="/api/v1")

    return app


app = create_app()
