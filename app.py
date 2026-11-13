"""Local FastAPI web application for confidence-thresholded AI4Mars image segmentation."""

from __future__ import annotations

import io
import random
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from inference import DatasetBrowser, SegmentationEngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AI4Mars SegFormer Lab", version="1.0.0")
engine = SegmentationEngine(BASE_DIR)
dataset = DatasetBrowser(BASE_DIR)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def get_config() -> dict:
    return {
        "model": engine.get_public_config(),
        "dataset": {
            "available": dataset.available,
            "counts": dataset.split_counts() if dataset.available else {},
        },
    }


@app.get("/api/dataset")
def list_dataset(
    split: str = Query("test"),
    search: str = Query(""),
    offset: int = Query(0, ge=0),
    limit: int = Query(60, ge=1, le=200),
) -> dict:
    if not dataset.available:
        return {"total": 0, "counts": {}, "items": []}
    return dataset.query(split=split, search=search, offset=offset, limit=limit)


@app.get("/api/dataset/random")
def random_sample(split: str = Query("test")) -> dict:
    if not dataset.available:
        raise HTTPException(status_code=404, detail="Dataset zip or CSV not found.")
    candidates = [e for e in dataset.entries if split == "all" or e.split == split]
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No samples found in split '{split}'.")
    chosen = random.choice(candidates)
    return _predict_dataset_entry(chosen.index)


@app.get("/api/dataset/sample/{index}")
def predict_sample(index: int) -> dict:
    if not dataset.available:
        raise HTTPException(status_code=404, detail="Dataset zip or CSV not found.")
    return _predict_dataset_entry(index)


def _predict_dataset_entry(index: int) -> dict:
    try:
        entry, img, _ = dataset.load_sample(index)
    except IndexError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = engine.predict(img)
    _, _, gt_data_url = dataset.load_sample(index, target_size=(result["width"], result["height"]))
    result["source"] = {
        "type": "dataset",
        "index": entry.index,
        "split": entry.split,
        "filename": entry.filename,
        "labeled_frac": entry.labeled_frac,
        "is_flipped": entry.is_flipped,
        "is_brightened": entry.is_brightened,
    }
    result["ground_truth_data_url"] = gt_data_url
    return result


@app.post("/api/predict")
async def predict_upload(file: UploadFile = File(...)) -> dict:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Unsupported or corrupted image file.") from exc

    result = engine.predict(img)
    result["source"] = {
        "type": "upload",
        "filename": file.filename or "uploaded_image",
    }
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
