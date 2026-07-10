"""
Basic unit + integration tests. Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src import preprocessing, layout_detection, pipeline

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "samples", "sample_invoice.png")


def test_preprocess_returns_expected_keys():
    result = preprocessing.preprocess(SAMPLE)
    assert set(result.keys()) == {"raw", "gray", "binary", "skew_angle"}
    assert result["gray"].ndim == 2
    assert result["binary"].ndim == 2


def test_skew_correction_reduces_angle():
    result = preprocessing.preprocess(SAMPLE)
    # after correction, re-estimating skew on the corrected image should be near zero
    residual = preprocessing.estimate_skew_angle(result["gray"])
    assert abs(residual) < 3.0


def test_layout_detects_at_least_one_table():
    pre = preprocessing.preprocess(SAMPLE)
    regions = layout_detection.analyze_layout(pre)
    tables = [r for r in regions if r.region_type == "table"]
    assert len(tables) >= 1


def test_layout_detects_multiple_text_blocks():
    pre = preprocessing.preprocess(SAMPLE)
    regions = layout_detection.analyze_layout(pre)
    text_blocks = [r for r in regions if r.region_type == "text"]
    assert len(text_blocks) >= 2


def test_full_pipeline_extracts_known_text():
    result = pipeline.run_pipeline(SAMPLE)
    all_text = " ".join(r["text"] for r in result["regions"]).lower()
    assert "invoice" in all_text
    assert "syringes" in all_text or "gauze" in all_text


def test_full_pipeline_confidence_reasonable():
    result = pipeline.run_pipeline(SAMPLE)
    confidences = [r["avg_confidence"] for r in result["regions"] if r["text"]]
    assert all(c > 0 for c in confidences)
    assert sum(confidences) / len(confidences) > 70


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
