# DocStructure

**Layout-aware document OCR: detect regions, then extract text — not the other way around.**

Most OCR demos run Tesseract over a whole page and call it done. That falls apart on real
documents — invoices, forms, prescriptions — where tables, headers, and paragraphs need to be
read *as structured regions*, not one undifferentiated block of text. DocStructure detects
layout first (text blocks vs. tables), then runs OCR per region, so the output is structured
data you can actually build on top of — not a wall of jumbled text.

This mirrors the approach used in production document-intelligence pipelines (invoice
extraction, prescription digitization, form processing).

## Pipeline

```
  Input Image
      │
      ▼
┌─────────────────┐    Denoise (Non-Local Means) + Otsu-based skew
│  Preprocessing   │    estimation + affine deskew + adaptive
└────────┬─────────┘    thresholding
         ▼
┌─────────────────┐    Morphological line detection (horiz/vert kernels)
│ Layout Detection │    → tables. Character→word→line→paragraph
└────────┬─────────┘    dilation → text blocks. Reading-order sort.
         ▼
┌─────────────────┐    Tesseract per region. Table crops have grid
│   OCR Engine     │    lines stripped first (biggest single accuracy
└────────┬─────────┘    win — see Results). Per-word confidence scoring.
         ▼
┌─────────────────┐    Region type, bbox, text, confidence,
│ Structured JSON  │    low-confidence flag for human review queue
└─────────────────┘
```

No pretrained layout model, no GPU — everything is classical CV (morphological ops,
connected components, projection-style grouping) plus Tesseract. That's a deliberate choice:
it runs anywhere, has zero model download/licensing overhead, and the failure modes are
fully inspectable instead of hidden inside a black-box detector.

## Example

Input (synthetic invoice, generated with random skew + scan noise to simulate a real photo):

`samples/sample_invoice.png`

Detected regions (orange = text, green = table):

`samples/sample_invoice_regions.png`

Output (`outputs/sample_result.json`, abbreviated):

```json
{
  "skew_angle_corrected": -0.57,
  "num_regions": 4,
  "num_tables": 1,
  "num_text_blocks": 3,
  "num_flagged_low_confidence": 0,
  "regions": [
    { "type": "text",  "avg_confidence": 90.9, "text": "ACME MEDICAL SUPPLIES Invoice #INV-2026—0731 Date: 10 July 2026" },
    { "type": "text",  "avg_confidence": 93.9, "text": "This invoice covers the supply of medical consumables delivered to..." },
    { "type": "table", "avg_confidence": 91.8, "text": "Item Qty Price Surgical Gloves (box) 50 450.00 Syringes 5ml 200 1200.00 ..." },
    { "type": "text",  "avg_confidence": 94.4, "text": "Total Amount: Rs, 3410.00 Thank you for your business." }
  ]
}
```

### Why the table-line stripping step matters

Stripping grid lines from table crops before OCR (`ocr_engine._strip_grid_lines`) took the
table region's average confidence from **12.3% → 91.8%** on the sample document — border
lines were being misread as characters. This is the kind of thing that's invisible in a demo
GIF but is exactly the difference between a pipeline that works on toy inputs and one that
holds up on real scans.

## Quickstart

```bash
# System dependency
sudo apt-get install tesseract-ocr

# Python dependencies
pip install -r requirements.txt

# Generate a synthetic test document (or use your own image)
python3 tests/generate_sample.py

# Run the pipeline
python3 -m src.pipeline samples/sample_invoice.png --out outputs/result.json

# Visualize detected regions
python3 tests/visualize_regions.py

# Run tests
pytest tests/ -v

# Serve as an API
uvicorn src.api:app --reload --port 8000
# then: curl -X POST http://localhost:8000/extract -F "file=@samples/sample_invoice.png"
```

## Design notes

- **Confidence-based review flagging**: every region below 60% average OCR confidence is
  flagged (`low_confidence: true`) rather than silently returned — the pattern used in
  production pipelines to route uncertain extractions to a human-in-the-loop queue instead of
  quietly shipping wrong data.
- **Tables detected before text**: table regions are found first (via their distinct grid-line
  signature) and masked out before text-block detection runs, avoiding double-detection of the
  same area as both a table and a paragraph.
- **Deskew is conservative**: corrections beyond 20° are skipped, since at that point it's more
  likely the estimator latched onto noise than an actual scan skew.

## What I'd add next

- Cell-level table parsing (currently table cells are OCR'd as one block; the grid coordinates
  from `layout_detection.detect_tables` are already available to split further)
- Swap the classical-CV layout detector for a fine-tuned lightweight detector (e.g. YOLOv8-n)
  trained on DocLayNet/PubLayNet for handling more visually complex layouts
- Multi-language support beyond Tesseract's `lang` parameter (e.g. TrOCR for handwritten text)

## Stack

Python · OpenCV · Tesseract OCR · FastAPI · pytest

---

*Built as a demonstration of layout-aware document processing — the same problem space as
production invoice/prescription/form-digitization systems, implemented end-to-end with
classical CV rather than opaque pretrained detectors.*
