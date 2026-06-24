"""
Race-start detection.

Tries to pin down t=0 (the actual start trigger) as precisely as possible,
in priority order from most to least reliable:

  1. Gun-shot audio transient   — most precise when the video has usable audio.
  2. Broadcast on-screen timer  — needs a manually-specified ROI of the clock.
  3. First sustained movement   — last resort, frame-rate limited.

Every method returns a StartDetectionResult (or None if it couldn't find
anything), so the caller can see WHICH method succeeded and how confident
it is, rather than silently trusting a number with no provenance.

Wire this in BEFORE you start computing speeds/splits — once you have a
start_timestamp, subtract it from every later frame timestamp so t=0
lines up with the actual start, not just "frame 0 of the video file".
"""

import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:
    import pytesseract
    _HAS_TESSERACT = True
except ImportError:
    _HAS_TESSERACT = False


@dataclass
class StartDetectionResult:
    start_timestamp: float
    method: str          # "gun_audio" | "broadcast_timer" | "first_movement"
    confidence: float     # rough 0-1 heuristic confidence, not a statistical guarantee
    details: str
    candidates: list = field(default_factory=list)  # raw scoring info, for debugging


# ---------------------------------------------------------------------------
# Method 1: gun-shot audio transient
# ---------------------------------------------------------------------------

def _extract_audio_track(video_path: str, sample_rate: int = 22050,
                          duration_s: Optional[float] = None) -> np.ndarray:
    """
    Pulls a mono float32 audio track out of the video via ffmpeg.
    Raises if ffmpeg isn't installed or the video has no audio stream —
    caller (detect_gun_shot) catches this and treats it as "method unavailable".
    """
    cmd = ["ffmpeg", "-i", video_path]
    if duration_s:
        cmd += ["-t", str(duration_s)]
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("ffmpeg produced no audio output (no audio track, or ffmpeg missing)")

    return np.frombuffer(proc.stdout, dtype=np.float32)


def detect_gun_shot(video_path: str, search_window_s: float = 15.0,
                     sample_rate: int = 22050, window_ms: float = 5.0,
                     baseline_s: float = 1.0, spike_ratio_threshold: float = 6.0
                     ) -> Optional[StartDetectionResult]:
    """
    Looks for a sharp transient in the first `search_window_s` seconds of audio.

    Gun shots are a sudden broadband spike: RMS energy jumps WAY above the
    quiet baseline that preceded it, almost instantly. We approximate this
    with a short-window RMS envelope and a rolling local-median baseline —
    no need for full spectral analysis, the amplitude jump alone is a strong
    enough signal for a starting gun specifically (as opposed to e.g. music,
    which ramps up rather than spiking).

    Returns None (not an exception) if there's no audio, ffmpeg is missing,
    or nothing spikes hard enough to pass the threshold — all of these mean
    "this method can't help here," which the caller should treat the same way.
    """
    try:
        audio = _extract_audio_track(video_path, sample_rate=sample_rate,
                                      duration_s=search_window_s)
    except Exception:
        return None

    if audio.size == 0:
        return None

    window = max(1, int(sample_rate * window_ms / 1000))
    n_windows = len(audio) // window
    if n_windows < 10:
        return None

    rms = np.sqrt(
        np.mean(audio[: n_windows * window].reshape(n_windows, window) ** 2, axis=1) + 1e-12
    )
    times = np.arange(n_windows) * (window / sample_rate)

    baseline_windows = max(1, int(baseline_s * sample_rate / window))

    best_idx = None
    best_ratio = 0.0
    for i in range(baseline_windows, n_windows):
        baseline = np.median(rms[max(0, i - baseline_windows):i]) + 1e-9
        ratio = rms[i] / baseline
        if ratio > best_ratio:
            best_ratio = ratio
            best_idx = i

    if best_idx is None or best_ratio < spike_ratio_threshold:
        return None

    confidence = float(min(1.0, 0.3 + (best_ratio - spike_ratio_threshold) / spike_ratio_threshold))

    return StartDetectionResult(
        start_timestamp=float(times[best_idx]),
        method="gun_audio",
        confidence=confidence,
        details=f"audio transient {best_ratio:.1f}x baseline RMS at t={times[best_idx]:.3f}s",
    )


# ---------------------------------------------------------------------------
# Method 2: on-screen broadcast race clock (OCR)
# ---------------------------------------------------------------------------

def _parse_clock_text(text: str) -> Optional[float]:
    """Best-effort parse of OCR'd clock text like '0.00', '00:00', '3.45' into seconds."""
    text = text.strip().replace("O", "0").replace(",", ".")
    if not text:
        return None

    m = re.match(r"^(\d{1,2})[:.](\d{1,3})$", text)
    if not m:
        return None
    whole, frac = m.groups()
    try:
        # sprint broadcast clocks are seconds.hundredths — treat frac as that
        divisor = 100 if len(frac) <= 2 else 1000
        return float(whole) + float(frac) / divisor
    except ValueError:
        return None


def _find_clock_start(readings: list, min_consecutive_increasing: int = 5,
                       jitter_tolerance: float = 0.02) -> Optional[int]:
    """
    Finds the LAST frame still showing the static pre-race value (e.g. "0.00"),
    immediately before a sustained run of increasing readings — i.e. the frame
    right before the clock visibly starts counting up.

    Uses frame-to-frame deltas rather than an absolute "near zero" threshold.
    An absolute threshold doesn't work here: a real broadcast clock increments
    at real-time speed (~0.033s per frame at 30fps), so it stays under almost
    any reasonable absolute cutoff for the first several frames after the gun
    too — which would bias the detected start time late. Comparing each frame
    to the one before it sidesteps that entirely.
    """
    values = [r[2] for r in readings]

    for i in range(1, len(values) - min_consecutive_increasing):
        if values[i] is None or values[i - 1] is None:
            continue
        if values[i] - values[i - 1] <= jitter_tolerance:
            continue  # not a real increase yet — could just be OCR jitter

        window = values[i: i + min_consecutive_increasing]
        if any(v is None for v in window):
            continue
        if all(window[j] < window[j + 1] for j in range(len(window) - 1)):
            return i - 1  # last frame still at the static pre-race value

    return None


def pick_roi_interactive(frame: np.ndarray,
                          window_name: str = "Click top-left then bottom-right of the clock, then press q"
                          ) -> Optional[tuple]:
    """
    Two-click rectangle picker for selecting the on-screen timer region.
    Click the top-left corner, then the bottom-right corner, then press 'q'.
    Returns (x1, y1, x2, y2) or None if cancelled before two points were clicked.
    """
    points = []
    display = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.rectangle(display, points[0], points[1], (0, 255, 0), 2)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(points) >= 2:
            break

    cv2.destroyAllWindows()
    if len(points) < 2:
        return None

    (x1, y1), (x2, y2) = points
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def detect_broadcast_timer_start(video_path: str, timer_roi: tuple,
                                  search_window_s: float = 15.0
                                  ) -> Optional[StartDetectionResult]:
    """
    timer_roi: (x1, y1, x2, y2) pixel box around the on-screen race clock,
    measured on the same frame you'd use for homography calibration.

    Crops that region every frame for the first `search_window_s` seconds,
    OCRs it, and looks for the moment it flips from "0.00" to counting up.

    Returns None if pytesseract isn't installed, or no clear start transition
    is found (wrong ROI, clock not actually visible in this window, etc).
    """
    if not _HAS_TESSERACT:
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    x1, y1, x2, y2 = [int(v) for v in timer_roi]

    readings = []
    frame_idx = 0
    max_frames = int(search_window_s * fps)

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        crop = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        text = pytesseract.image_to_string(
            thresh, config="--psm 7 -c tessedit_char_whitelist=0123456789.:"
        ).strip()
        readings.append((frame_idx, frame_idx / fps, _parse_clock_text(text)))
        frame_idx += 1

    cap.release()

    start_idx = _find_clock_start(readings)
    if start_idx is None:
        return None

    frame_idx, ts, value = readings[start_idx]
    return StartDetectionResult(
        start_timestamp=ts,
        method="broadcast_timer",
        confidence=0.6,
        details=f"clock read ~{value:.2f}s at frame {frame_idx}, then began incrementing",
    )


# ---------------------------------------------------------------------------
# Method 3: first sustained movement (fallback, frame-rate limited)
# ---------------------------------------------------------------------------

def detect_first_movement(position_histories: dict, movement_threshold_mps: float = 0.5,
                           sustained_frames: int = 3) -> Optional[StartDetectionResult]:
    """
    position_histories: {track_id: [(frame_idx, timestamp, world_x, world_y), ...]}
    (i.e. TrackedAthlete.position_history for each athlete, from the opening
    seconds of the race.)

    Finds, per athlete, the first timestamp where forward speed exceeds
    `movement_threshold_mps` and STAYS above it for `sustained_frames` in a
    row — filtering out a single noisy jitter frame counting as "movement".
    Returns the earliest such timestamp across all athletes.

    THIS IS A FALLBACK. It's limited by frame rate (≈33ms buckets at 30fps),
    and a clean reaction-time race can be decided within that window. Only
    use this when gun_audio and broadcast_timer both come back None, and
    treat the confidence score as a flag to surface in your UI, not hide.
    """
    earliest = None

    for tid, hist in position_histories.items():
        hist_sorted = sorted(hist, key=lambda h: h[0])
        for i in range(1, len(hist_sorted) - sustained_frames):
            speeds = []
            ok = True
            for j in range(sustained_frames):
                a, b = hist_sorted[i - 1 + j], hist_sorted[i + j]
                dt = b[1] - a[1]
                if dt <= 0:
                    ok = False
                    break
                speeds.append(abs(b[2] - a[2]) / dt)
            if ok and all(s > movement_threshold_mps for s in speeds):
                ts = hist_sorted[i - 1][1]
                if earliest is None or ts < earliest[0]:
                    earliest = (ts, tid)
                break

    if earliest is None:
        return None

    ts, tid = earliest
    return StartDetectionResult(
        start_timestamp=ts,
        method="first_movement",
        confidence=0.35,
        details=(f"earliest sustained forward motion: athlete {tid} at t={ts:.3f}s "
                 f"(frame-rate limited estimate — not a precise gun time)"),
    )


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def detect_race_start(video_path: str, position_histories: Optional[dict] = None,
                       timer_roi: Optional[tuple] = None) -> StartDetectionResult:
    """
    Tries each method in priority order, returns the first that succeeds.
    Raises ValueError if every available method fails, rather than silently
    defaulting to frame 0 — you want to know when this happens, not have
    every downstream split time be quietly wrong.

    position_histories and timer_roi are both optional because you may not
    have either available yet (e.g. you haven't run tracking on the opening
    frames, or you haven't picked a clock ROI) — pass whatever you've got.
    """
    gun_result = detect_gun_shot(video_path)
    if gun_result:
        return gun_result

    if timer_roi is not None:
        timer_result = detect_broadcast_timer_start(video_path, timer_roi)
        if timer_result:
            return timer_result

    if position_histories is not None:
        movement_result = detect_first_movement(position_histories)
        if movement_result:
            return movement_result

    raise ValueError(
        "Could not detect race start via gun audio, broadcast timer, or first "
        "movement. Provide a manual start_timestamp, or pass timer_roi / "
        "position_histories to enable those detection methods."
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python start_detector.py <video_path>")
    else:
        video_path = sys.argv[1]
        result = detect_gun_shot(video_path)
        if result:
            print(f"Gun shot detected: {result}")
        else:
            print("No gun shot detected via audio (no audio track, ffmpeg missing, "
                  "or no transient passed the threshold). Try broadcast_timer or "
                  "first_movement instead.")