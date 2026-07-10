"""
pipeline.py
Orchestrates preprocessing -> layout detection -> OCR -> structured output.
This is the single entry point used by both the CLI demo and the API.
"""
import json
import time
from . import preprocessing, layout_detection, ocr_engine


def run_pipeline(image_path: str, lang: str = "eng", debug_prefix: str = None) -> dict:
    t0 = time.time()

    pre = preprocessing.preprocess(image_path, save_debug_to=debug_prefix)
    regions = layout_detection.analyze_layout(pre)
    ocr_results = ocr_engine.ocr_all_regions(pre["gray"], regions, lang=lang)

    n_flagged = sum(1 for r in ocr_results if r.get("low_confidence"))

    output = {
        "source_image": image_path,
        "skew_angle_corrected": round(pre["skew_angle"], 2),
        "num_regions": len(ocr_results),
        "num_tables": sum(1 for r in ocr_results if r["type"] == "table"),
        "num_text_blocks": sum(1 for r in ocr_results if r["type"] == "text"),
        "num_flagged_low_confidence": n_flagged,
        "processing_time_sec": round(time.time() - t0, 3),
        "regions": ocr_results,
    }
    return output


def run_and_save(image_path: str, out_json: str, lang: str = "eng"):
    result = run_pipeline(image_path, lang=lang)
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DocStructure: layout-aware document OCR")
    parser.add_argument("image", help="Path to input document image")
    parser.add_argument("--out", default="output.json", help="Path to output JSON")
    parser.add_argument("--lang", default="eng", help="Tesseract language code")
    args = parser.parse_args()

    result = run_and_save(args.image, args.out, lang=args.lang)
    print(f"Processed {args.image}")
    print(f"  Regions found: {result['num_regions']} "
          f"({result['num_tables']} tables, {result['num_text_blocks']} text blocks)")
    print(f"  Flagged for review (low OCR confidence): {result['num_flagged_low_confidence']}")
    print(f"  Time: {result['processing_time_sec']}s")
    print(f"  Output saved to {args.out}")
