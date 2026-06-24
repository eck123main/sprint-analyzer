"""
Synthetic tests for start_detector.py — no real video/audio needed.
Run with: python -m pytest tests/test_start_detector.py -v
(or just: python tests/test_start_detector.py)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../backend/utils"))

from start_detector import _parse_clock_text, _find_clock_start, detect_first_movement


def test_parse_clock_text():
    assert _parse_clock_text("0.00") == 0.0
    assert _parse_clock_text("3.45") == 3.45
    assert _parse_clock_text("00:12") == 0.12
    assert _parse_clock_text("garbage") is None
    assert _parse_clock_text("") is None
    # OCR commonly misreads 0 as O
    assert _parse_clock_text("O.OO") == 0.0


def test_find_clock_start_detects_transition():
    # frames 0-4 sit at ~0.00, then frames 5+ count up — start should land at index 4
    readings = [(i, i / 30.0, 0.0) for i in range(5)]
    readings += [(i, i / 30.0, (i - 4) * 0.033) for i in range(5, 12)]
    start_idx = _find_clock_start(readings)
    assert start_idx == 4, f"expected last near-zero frame (4), got {start_idx}"


def test_find_clock_start_no_transition_returns_none():
    # clock never moves — nothing to find
    readings = [(i, i / 30.0, 0.0) for i in range(20)]
    assert _find_clock_start(readings) is None


def test_detect_first_movement_picks_earliest_mover():
    fps = 30.0
    # Athlete 1 starts moving at frame 10, athlete 2 at frame 20 — expect athlete 1's time
    hist_1 = [(i, i / fps, 0.0 if i < 10 else (i - 10) * 1.0, 0.0) for i in range(40)]
    hist_2 = [(i, i / fps, 0.0 if i < 20 else (i - 20) * 1.0, 1.22) for i in range(40)]

    result = detect_first_movement({1: hist_1, 2: hist_2})

    assert result is not None
    assert result.method == "first_movement"
    # athlete 1 should be the earliest mover, around frame 10 (t ≈ 0.33s)
    assert abs(result.start_timestamp - (10 / fps)) < 0.05


def test_detect_first_movement_no_movement_returns_none():
    fps = 30.0
    # nobody moves the whole time
    hist = [(i, i / fps, 0.0, 0.0) for i in range(40)]
    result = detect_first_movement({1: hist})
    assert result is None


if __name__ == "__main__":
    tests = [
        test_parse_clock_text,
        test_find_clock_start_detects_transition,
        test_find_clock_start_no_transition_returns_none,
        test_detect_first_movement_picks_earliest_mover,
        test_detect_first_movement_no_movement_returns_none,
    ]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")