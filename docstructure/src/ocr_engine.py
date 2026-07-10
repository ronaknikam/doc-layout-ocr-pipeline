"""
ocr_engine.py
Runs Tesseract OCR on individual layout regions and returns text with
per-word confidence, so low-confidence extractions can be flagged for
human review (same pattern used in production document-AI pipelines).
"""
import cv2
import pytesseract
import numpy as np
from typing import List
from .layout_detection import Region

PAD = 4  # small padding around each region improves OCR accuracy at edges


def _crop(gray: np.ndarray, region: Region) -> np.ndarray:
    h, w = gray.shape[:2]
    x0 = max(region.x - PAD, 0)
    y0 = max(region.y - PAD, 0)
    x1 = min(region.x + region.w + PAD, w)
    y1 = min(region.y + region.h + PAD, h)
    return gray[y0:y1, x0:x1]


def ocr_region(gray: np.ndarray, region: Region, lang: str = "eng") -> dict:
    crop = _crop(gray, region)
    if crop.size == 0:
        return {"text": "", "avg_confidence": 0.0, "low_confidence": True}

    config = "--oem 3 --psm 6"  # PSM 6: assume a uniform block of text
    data = pytesseract.image_to_data(
        crop, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    words, confidences = [], []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        conf = float(conf)
        if text and conf >= 0:
            words.append(text)
            confidences.append(conf)

    full_text = " ".join(words)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "text": full_text,
        "avg_confidence": round(avg_conf, 1),
        "low_confidence": avg_conf < 60.0,  # flag for human review
        "word_count": len(words),
    }


def ocr_all_regions(gray: np.ndarray, regions: List[Region], lang: str = "eng") -> List[dict]:
    results = []
    for region in regions:
        if region.region_type == "table":
            ocr_result = ocr_table_region(gray, region, lang=lang)
        else:
            ocr_result = ocr_region(gray, region, lang=lang)
        results.append({
            "type": region.region_type,
            "bbox": region.bbox,
            **ocr_result,
        })
    return results


def _strip_grid_lines(crop: np.ndarray) -> np.ndarray:
    """
    Table borders confuse Tesseract badly (grid lines get read as garbage
    characters). We detect long horizontal/vertical lines the same way
    layout_detection does and paint them white before OCR, leaving just
    the cell text behind.
    """
    thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    h, w = thresh.shape

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 20, 10), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 20, 10)))
    horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)
    grid = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))

    cleaned = crop.copy()
    cleaned[grid > 0] = 255
    return cleaned


def ocr_table_region(gray: np.ndarray, region: Region, lang: str = "eng") -> dict:
    """
    Strips grid lines first, then OCRs the cleaned crop. Callers wanting
    cell-level (not just region-level) extraction can further split the
    cleaned crop along the same grid coordinates -- that hook lives here
    so it's a single place to extend for structured table parsing.
    """
    crop = _crop(gray, region)
    if crop.size == 0:
        return {"text": "", "avg_confidence": 0.0, "low_confidence": True}

    cleaned = _strip_grid_lines(crop)
    config = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(
        cleaned, lang=lang, config=config, output_type=pytesseract.Output.DICT
    )

    words, confidences = [], []
    for text, conf in zip(data["text"], data["conf"]):
        text = text.strip()
        conf = float(conf)
        if text and conf >= 0:
            words.append(text)
            confidences.append(conf)

    full_text = " ".join(words)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "text": full_text,
        "avg_confidence": round(avg_conf, 1),
        "low_confidence": avg_conf < 60.0,
        "word_count": len(words),
    }
