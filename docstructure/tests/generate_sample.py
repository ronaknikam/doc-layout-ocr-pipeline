"""Generates a synthetic invoice-like document image (with a title,
paragraph text, and a bordered table) to validate the pipeline end-to-end
without needing external test data."""
import cv2
import numpy as np
import random

W, H = 1000, 1300
img = np.ones((H, W, 3), dtype=np.uint8) * 255
font = cv2.FONT_HERSHEY_SIMPLEX

cv2.putText(img, "ACME MEDICAL SUPPLIES", (60, 80), font, 1.1, (0, 0, 0), 2)
cv2.putText(img, "Invoice #INV-2026-0731", (60, 120), font, 0.7, (0, 0, 0), 1)
cv2.putText(img, "Date: 10 July 2026", (60, 150), font, 0.7, (0, 0, 0), 1)

paragraph = [
    "This invoice covers the supply of medical consumables",
    "delivered to Pune General Hospital under purchase order",
    "PO-44231. Payment is due within 30 days of receipt.",
]
for i, line in enumerate(paragraph):
    cv2.putText(img, line, (60, 220 + i * 35), font, 0.6, (0, 0, 0), 1)

# Table
tx, ty, tw, th = 60, 380, 880, 320
rows, cols = 5, 3
cv2.rectangle(img, (tx, ty), (tx + tw, ty + th), (0, 0, 0), 2)
row_h = th // rows
col_w = tw // cols
for r in range(1, rows):
    cv2.line(img, (tx, ty + r * row_h), (tx + tw, ty + r * row_h), (0, 0, 0), 1)
for c in range(1, cols):
    cv2.line(img, (tx + c * col_w, ty), (tx + c * col_w, ty + th), (0, 0, 0), 1)

headers = ["Item", "Qty", "Price"]
rows_data = [
    ["Surgical Gloves (box)", "50", "450.00"],
    ["Syringes 5ml", "200", "1200.00"],
    ["Gauze Rolls", "100", "800.00"],
    ["Sanitizer 500ml", "80", "960.00"],
]
for c, h_text in enumerate(headers):
    cv2.putText(img, h_text, (tx + c * col_w + 10, ty + 30), font, 0.6, (0, 0, 0), 2)
for r, row in enumerate(rows_data):
    for c, cell in enumerate(row):
        cv2.putText(img, cell, (tx + c * col_w + 10, ty + (r + 1) * row_h + 30),
                    font, 0.55, (0, 0, 0), 1)

footer = [
    "Total Amount: Rs. 3410.00",
    "Thank you for your business.",
]
for i, line in enumerate(footer):
    cv2.putText(img, line, (60, 760 + i * 35), font, 0.65, (0, 0, 0), 1)

# Apply a slight random rotation to simulate a real scanned skew
angle = random.uniform(-2.5, 2.5)
M = cv2.getRotationMatrix2D((W // 2, H // 2), angle, 1.0)
img = cv2.warpAffine(img, M, (W, H), borderValue=(255, 255, 255))

# Add mild Gaussian noise to simulate scan artifacts
noise = np.random.normal(0, 6, img.shape).astype(np.int16)
img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

cv2.imwrite("/home/claude/docstructure/samples/sample_invoice.png", img)
print(f"Sample generated with skew angle: {angle:.2f} degrees")
