import sys
from pathlib import Path

# Add project root directory to sys.path to support running this file directly with python
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.exceptions import ProviderError, MediaKitException
from app.routers.image_router import router as image_router

app = FastAPI(
    title=settings.APP_NAME,
    description="MVP for local/offline image generation capabilities.",
    version="1.0.0",
    debug=settings.DEBUG
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom exception handlers
@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    return JSONResponse(
        status_code=502, # Bad Gateway (upstream model failure)
        content={
            "error": "ProviderError",
            "provider": exc.provider_name,
            "message": exc.message
        }
    )

@app.exception_handler(MediaKitException)
async def mediakit_error_handler(request: Request, exc: MediaKitException):
    return JSONResponse(
        status_code=400,
        content={
            "error": "MediaKitException",
            "message": exc.message
        }
    )

# Startup event to initialize the active provider
@app.on_event("startup")
async def startup_event():
    from app.services.image_service import ImageService
    service = ImageService()
    await service.initialize()

# Include API routers
app.include_router(image_router, prefix="/api/v1")

@app.get("/", tags=["health"])
async def root():
    return {
        "app": settings.APP_NAME,
        "env": settings.APP_ENV,
        "status": "healthy",
        "provider": settings.IMAGE_PROVIDER
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
