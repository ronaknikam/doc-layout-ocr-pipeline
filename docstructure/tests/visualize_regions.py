"""Draws detected region bounding boxes on the source image -- useful for
README screenshots and for visually debugging layout detection."""
import sys
import cv2
sys.path.insert(0, "/home/claude/docstructure")
from src import preprocessing, layout_detection

COLORS = {"text": (255, 140, 0), "table": (0, 160, 0), "figure": (0, 0, 220)}


def visualize(image_path: str, out_path: str):
    pre = preprocessing.preprocess(image_path)
    regions = layout_detection.analyze_layout(pre)

    canvas = cv2.cvtColor(pre["gray"], cv2.COLOR_GRAY2BGR)
    for r in regions:
        color = COLORS.get(r.region_type, (128, 128, 128))
        cv2.rectangle(canvas, (r.x, r.y), (r.x + r.w, r.y + r.h), color, 3)
        cv2.putText(canvas, r.region_type, (r.x, max(r.y - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(out_path, canvas)
    print(f"Saved visualization to {out_path} ({len(regions)} regions)")


if __name__ == "__main__":
    visualize("/home/claude/docstructure/samples/sample_invoice.png",
               "/home/claude/docstructure/samples/sample_invoice_regions.png")
