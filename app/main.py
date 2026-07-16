from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.services.model_service import (
    InvalidImageError,
    ModelNotReadyError,
    ModelService,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
model_service = ModelService(settings.model_path, settings.class_names)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(model_service.load)
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="MobileNetV2 brain MRI classification with Grad-CAM explainability.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    status = model_service.status()
    return {
        "status": "ok" if status.ready else "degraded",
        "model_ready": status.ready,
        "message": status.message,
        "input_size": list(status.input_size),
        "classes": status.classes,
    }


@app.get("/api/model-info")
def model_info() -> dict:
    status = model_service.status()
    return {
        "ready": status.ready,
        "message": status.message,
        "input_size": list(status.input_size),
        "classes": status.classes,
        "architecture": "MobileNetV2 + GlobalAveragePooling2D + Dropout(0.3) + Softmax",
        "reported_test_accuracy": 87.81,
        "research_only": True,
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/tiff",
    }
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a PNG, JPEG, WEBP, BMP, or TIFF image.",
        )

    raw_bytes = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw_bytes) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"The image exceeds the {settings.max_upload_mb} MB limit.",
        )

    try:
        return await run_in_threadpool(model_service.predict, raw_bytes)
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check the server log for details.",
        ) from exc
