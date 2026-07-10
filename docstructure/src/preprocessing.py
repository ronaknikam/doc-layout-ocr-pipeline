"""
preprocessing.py
Classical CV preprocessing for scanned/photographed documents:
deskewing, denoising, and adaptive binarization.
"""
import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """
    Estimate document skew using the minAreaRect of all foreground pixels
    after Otsu thresholding. Robust to sparse text because we work on the
    binary mask rather than raw edges.
    """
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 20:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # cv2.minAreaRect returns angle in (-90, 0]; normalize to a small correction
    if angle < -45:
        angle = 90 + angle
    # Only correct small skews (avoid flipping documents that are intentionally rotated)
    if abs(angle) > 20:
        return 0.0
    return angle


def deskew(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.1:
        return gray
    (h, w) = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def adaptive_binarize(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=15,
    )


def preprocess(path: str, save_debug_to: str = None) -> dict:
    """
    Full preprocessing pipeline. Returns dict with intermediate + final images
    so callers/tests can inspect each stage.
    """
    raw = load_image(path)
    gray = to_grayscale(raw)
    denoised = denoise(gray)
    angle = estimate_skew_angle(denoised)
    deskewed = deskew(denoised, angle)
    binary = adaptive_binarize(deskewed)

    result = {
        "raw": raw,
        "gray": deskewed,       # deskewed grayscale, good for OCR
        "binary": binary,       # deskewed + binarized, good for layout detection
        "skew_angle": angle,
    }

    if save_debug_to:
        cv2.imwrite(f"{save_debug_to}_gray.png", result["gray"])
        cv2.imwrite(f"{save_debug_to}_binary.png", result["binary"])

    return result
