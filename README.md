# NeuroVision AI — deployable Keras + frontend

A full-stack research application for the group's MobileNetV2 brain-tumour MRI classifier. It provides:

- responsive academic-futuristic frontend;
- drag-and-drop MRI upload;
- four-class Keras prediction;
- confidence bars for every class;
- Grad-CAM explainability overlay;
- health endpoint and model metadata endpoint;
- Docker and cloud deployment configuration;
- upload type and size validation.

> **Research prototype:** This project is not a medical device and must not be used as a diagnosis.

## Project structure

```text
brain-tumor-mri-app/
├── app/
│   ├── main.py                 # FastAPI endpoints
│   ├── config.py               # Environment configuration
│   ├── services/model_service.py
│   └── static/                 # HTML, CSS, JavaScript and visual asset
├── models/
│   └── best_model.keras        # add your trained model here
├── scripts/check_model.py
├── tests/test_health.py
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── requirements.txt
├── MODEL_CARD.md
└── README.md
```

## 1. Export the trained model from Colab

Your notebook already saves the best checkpoint as `best_model.keras` through `ModelCheckpoint`. After the training cells have completed, run:

```python
from google.colab import files
files.download("best_model.keras")
```

Copy the downloaded file into:

```text
models/best_model.keras
```

The expected class order is:

```text
glioma,meningioma,notumor,pituitary
```

## 2. Run locally

Use Python 3.11 for the most predictable TensorFlow deployment environment.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_model.py
uvicorn app.main:app --reload
```

### macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_model.py
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

API documentation is available at `http://127.0.0.1:8000/docs`.

## 3. Run with Docker

```bash
docker build -t neurovision-ai .
docker run --rm -p 8000:8000 neurovision-ai
```

Or:

```bash
docker compose up --build
```

## 4. Deploy from GitHub

1. Place `best_model.keras` in `models/` before building, or store it in a private model registry/object store and copy it during deployment.
2. Push this project to a GitHub repository.
3. Create a Docker-based web service on your host.
4. Use `/health` as the health-check route.
5. Ensure the instance has enough RAM for TensorFlow and the MobileNetV2 model. A paid starter instance is safer than a very small free instance.

The included `render.yaml` can be used for a Render Blueprint deployment. Other Docker-compatible platforms can use the same `Dockerfile`.

## API

### `GET /health`

Returns server and model readiness.

### `GET /api/model-info`

Returns model architecture, class names, input size, and the recorded project test accuracy.

### `POST /api/predict`

Multipart field name: `file`

Accepted types: PNG, JPEG, WEBP, BMP, TIFF. Maximum size: 10 MB by default.

Example response:

```json
{
  "prediction": {
    "class_key": "pituitary",
    "label": "Pituitary Tumour",
    "confidence": 0.9734,
    "confidence_percentage": 97.34
  },
  "probabilities": [],
  "images": {
    "original": "data:image/png;base64,...",
    "gradcam": "data:image/png;base64,..."
  }
}
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/best_model.keras` | Location of the saved Keras model |
| `CLASS_NAMES` | `glioma,meningioma,notumor,pituitary` | Model output order |
| `MAX_UPLOAD_MB` | `10` | Maximum uploaded image size |
| `PORT` | `8000` | Server port in Docker/cloud |

## Production hardening checklist

Before handling real patient data, add authentication, encrypted transport, private storage, retention rules, malware scanning, audit logs, request throttling, monitoring, external-dataset validation, model versioning, and formal clinical/regulatory review. Do not store identifiable medical images in logs.
