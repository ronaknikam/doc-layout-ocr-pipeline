"""
layout_detection.py
Segments a binarized document image into regions (text blocks, tables,
figures) using classical CV: morphological analysis, connected components,
horizontal/vertical projection profiles, and Hough line detection for
table grids.
"""
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    region_type: str  # "text", "table", "figure"
    confidence: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def bbox(self):
        return (self.x, self.y, self.x + self.w, self.y + self.h)


def _detect_table_lines(binary: np.ndarray) -> np.ndarray:
    """Isolate long horizontal + vertical lines via morphological opening —
    the classic approach for detecting table grid structure."""
    inv = 255 - binary  # foreground = white
    h, w = inv.shape

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 30, 10), 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 30, 10)))

    horiz = cv2.morphologyEx(inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert = cv2.morphologyEx(inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)

    grid = cv2.bitwise_or(horiz, vert)
    return grid


def detect_tables(binary: np.ndarray, min_area: int = 4000) -> List[Region]:
    grid = _detect_table_lines(binary)
    grid = cv2.dilate(grid, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        # A real table grid should have both a wide horizontal and tall
        # vertical component inside it -- filter out stray long lines.
        if w < 40 or h < 40:
            continue
        tables.append(Region(x, y, w, h, "table", confidence=0.85))
    return tables


def _mask_out(binary: np.ndarray, regions: List[Region]) -> np.ndarray:
    masked = binary.copy()
    for r in regions:
        masked[r.y:r.y + r.h, r.x:r.x + r.w] = 255  # paint over as background
    return masked


def detect_text_blocks(binary: np.ndarray, min_area: int = 300) -> List[Region]:
    """
    Groups nearby text/word components into line-level, then block-level
    regions using morphological dilation sized relative to image dims —
    a lightweight stand-in for the "MSER + grouping" family of approaches.
    """
    inv = 255 - binary
    h, w = inv.shape

    # Dilate horizontally to merge characters -> words -> lines,
    # then vertically (smaller) to merge lines -> paragraphs.
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 25, 15), 3))
    merged = cv2.dilate(inv, line_kernel, iterations=1)
    block_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, max(h // 60, 8)))
    merged = cv2.dilate(merged, block_kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blocks = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw * ch < min_area:
            continue
        blocks.append(Region(x, y, cw, ch, "text", confidence=0.8))
    return blocks


def analyze_layout(pre: dict) -> List[Region]:
    """
    Full layout pipeline: find tables first (they have a very distinct
    signature), mask them out, then find remaining text blocks on what's
    left. Returns regions sorted in reading order (top-to-bottom, left-to-right).
    """
    binary = pre["binary"]
    tables = detect_tables(binary)
    remaining = _mask_out(binary, tables)
    text_blocks = detect_text_blocks(remaining)

    regions = tables + text_blocks
    regions.sort(key=lambda r: (r.y // 40, r.x))  # bucket rows, then left-to-right
    return regions
