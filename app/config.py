from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "NeuroVision AI")
    model_path: Path = Path(
        os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best_model.keras"))
    )
    class_names: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv(
            "CLASS_NAMES", "glioma,meningioma,notumor,pituitary"
        ).split(",")
        if item.strip()
    )
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))


settings = Settings()
