"""
api.py
FastAPI inference service. Run with:
    uvicorn src.api:app --reload --port 8000

POST an image to /extract and get back structured layout + OCR JSON.
"""
import shutil
import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from .pipeline import run_pipeline

app = FastAPI(
    title="DocStructure API",
    description="Layout-aware document OCR: detects text blocks and tables, "
                 "then extracts text per region with confidence scoring.",
    version="1.0.0",
)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile = File(...), lang: str = "eng"):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXT}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        result = run_pipeline(tmp_path, lang=lang)
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}")
    finally:
        os.remove(tmp_path)

    return result
