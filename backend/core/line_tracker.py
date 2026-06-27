"""
lane_tracker.py

Replaces homography.py + CameraMotionTracker for panning cameras.

Instead of calibrating once and correcting for camera motion, this detects
the white lane lines in EVERY frame and uses their pixel spacing to derive
a fresh pixels-per-metre scale each time. The camera can pan freely — as
long as at least two lane lines are visible, we have our ruler.

Key assumptions (valid for a standard athletics track):
- Lane width = 1.22m always
- Lane lines are white/light on a red/dark surface
- Camera pans smoothly (no jump cuts mid-race)
- Runner stays in one lane throughout
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.signal import savgol_filter


@dataclass
class LaneReading:
    frame_idx: int
    timestamp: float
    # Position along the track in metres (x), across in metres (y)
    world_x: float
    world_y: float
    # Pixels per metre derived from lane line spacing this frame
    px_per_m: float
    # How many lane lines were found (confidence proxy)
    lines_found: int


@dataclass 
class LaneCalibration:
    """
    Minimal calibration: just tell us where x=0 is (the start line)
    in pixel space on the first frame. Everything else is derived from
    lane line detection.
    """
    start_x_pixel: float   # pixel x of the start line on frame 0
    runner_lane: int        # which lane (1-8) the runner is in


class LaneLine:
    """A detected lane line — filtered and smoothed across frames."""
    def __init__(self, x_pixel: float):
        self.x_pixel = x_pixel  # current pixel x position of this line
        self._history = [x_pixel]
        self.ema_alpha = 0.3

    def update(self, x_pixel: float):
        # Exponential moving average to smooth jitter
        self.x_pixel = (1 - self.ema_alpha) * self.x_pixel + self.ema_alpha * x_pixel
        self._history.append(self.x_pixel)

    def velocity(self) -> float:
        """Pixels per frame the line is moving — tells us how fast camera pans."""
        if len(self._history) < 2:
            return 0.0
        return self._history[-1] - self._history[-2]


def detect_lane_lines(frame: np.ndarray,
                      min_line_length: int = 80,
                      max_line_gap: int = 30) -> list[float]:
    """
    Finds vertical-ish white lane lines in the frame.
    Returns sorted list of x pixel positions where lines were found.

    Works by:
    1. Converting to HSV and thresholding for white/light pixels
    2. Running Canny edge detection
    3. Using HoughLinesP to find line segments
    4. Filtering to near-vertical lines only (lane lines run away from camera)
    5. Clustering nearby lines into single detections
    """
    h, w = frame.shape[:2]

    # Focus on the bottom 60% of frame — lane lines are on the ground
    roi_top = int(h * 0.4)
    roi = frame[roi_top:, :]

    # Threshold for white/light colours (lane lines)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # White: low saturation, high value
    white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

    # Also catch slightly off-white lines
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(white_mask, bright_mask)

    # Clean up noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lines is None:
        return []

    # Filter to lines that are somewhat vertical (within 45 degrees of vertical)
    # Lane lines viewed from the side are actually diagonal in perspective,
    # so we accept a wide angle range
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) < 1:
            angle = 90.0
        else:
            angle = abs(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))))

        # Accept lines between 20 and 90 degrees from horizontal
        if angle > 20:
            # Use midpoint x as the line's position
            mid_x = (x1 + x2) / 2
            vertical_lines.append(mid_x)

    if not vertical_lines:
        return []

    # Cluster nearby detections — multiple segments on the same line
    # should count as one line
    vertical_lines.sort()
    clusters = []
    current_cluster = [vertical_lines[0]]

    for x in vertical_lines[1:]:
        if x - current_cluster[-1] < 40:  # within 40px = same line
            current_cluster.append(x)
        else:
            clusters.append(np.mean(current_cluster))
            current_cluster = [x]
    clusters.append(np.mean(current_cluster))

    return sorted(clusters)


def find_lane_scale(line_positions: list[float]) -> Optional[float]:
    """
    Given a list of detected lane line x-positions (pixels), find the
    most consistent spacing and return pixels-per-metre.

    Standard lane width = 1.22m, so if two adjacent lines are D pixels apart,
    pixels_per_metre = D / 1.22
    """
    if len(line_positions) < 2:
        return None

    spacings = []
    for i in range(len(line_positions) - 1):
        spacing = line_positions[i + 1] - line_positions[i]
        if spacing > 20:  # ignore tiny gaps (noise)
            spacings.append(spacing)

    if not spacings:
        return None

    # Use median spacing — robust to one bad detection
    median_spacing = np.median(spacings)

    # Sanity check: lane width in pixels should be reasonable
    # At typical track filming distance, expect 50-400px per lane
    if median_spacing < 20 or median_spacing > 500:
        return None

    return median_spacing / 1.22  # pixels per metre


class LaneTracker:
    """
    Main class. Call update() each frame with the frame and athlete
    pixel position. Returns world-space coordinates.

    On first call, detects lane lines and estimates start position.
    On subsequent calls, tracks line movement to follow camera pan
    and updates the scale accordingly.
    """

    def __init__(self, calibration: LaneCalibration):
        self.calibration = calibration
        self.lane_lines: list[LaneLine] = []
        self.px_per_m: Optional[float] = None
        self.cumulative_x_m: float = 0.0  # distance athlete has covered
        self._last_athlete_x_px: Optional[float] = None
        self._last_frame_idx: Optional[int] = None
        self._px_per_m_history: list[float] = []
        self.frames_since_detection = 0
        self.redetect_interval = 5  # re-run full detection every N frames

    def _update_lane_lines(self, detected_positions: list[float]):
        """Match detected lines to tracked lines, update or create."""
        if not self.lane_lines:
            self.lane_lines = [LaneLine(x) for x in detected_positions]
            return

        # Match each detection to nearest existing tracked line
        used_tracked = set()
        for det_x in detected_positions:
            best_dist = float('inf')
            best_idx = None
            for i, line in enumerate(self.lane_lines):
                if i in used_tracked:
                    continue
                dist = abs(line.x_pixel - det_x)
                if dist < best_dist and dist < 60:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None:
                self.lane_lines[best_idx].update(det_x)
                used_tracked.add(best_idx)
            else:
                # New line appeared (camera panned to reveal it)
                self.lane_lines.append(LaneLine(det_x))

        # Sort by x position
        self.lane_lines.sort(key=lambda l: l.x_pixel)

        # Drop lines that have drifted off screen
        self.lane_lines = [l for l in self.lane_lines
                           if 0 < l.x_pixel < 99999]

    def update(self, frame: np.ndarray, athlete_px: float,
               frame_idx: int, timestamp: float) -> Optional[LaneReading]:
        """
        frame: current video frame (BGR)
        athlete_px: x pixel position of athlete's hip in this frame
        frame_idx: frame number
        timestamp: seconds since race start

        Returns LaneReading with world coordinates, or None if not enough
        lane lines are visible to compute a reliable position.
        """
        # Re-detect lane lines periodically
        if self.frames_since_detection >= self.redetect_interval or not self.lane_lines:
            detected = detect_lane_lines(frame)
            self._update_lane_lines(detected)
            self.frames_since_detection = 0
        else:
            self.frames_since_detection += 1

        if len(self.lane_lines) < 2:
            return None  # can't compute scale without 2 lines

        line_positions = [l.x_pixel for l in self.lane_lines]
        px_per_m = find_lane_scale(line_positions)

        if px_per_m is None:
            return None

        # Smooth the scale estimate over time
        self._px_per_m_history.append(px_per_m)
        if len(self._px_per_m_history) > 30:
            self._px_per_m_history.pop(0)
        smooth_px_per_m = float(np.median(self._px_per_m_history))

        # Track how far the athlete has moved since last frame
        if self._last_athlete_x_px is not None and self._last_frame_idx is not None:
            # Pixel displacement of athlete
            athlete_px_delta = athlete_px - self._last_athlete_x_px

            # But the camera also panned — subtract camera motion
            # Camera motion = average movement of all tracked lane lines
            if len(self.lane_lines) >= 2:
                camera_pan_px = np.mean([l.velocity() for l in self.lane_lines])
            else:
                camera_pan_px = 0.0

            # True athlete movement = athlete pixel delta - camera pan
            true_movement_px = athlete_px_delta - camera_pan_px

            # Convert to metres
            movement_m = true_movement_px / smooth_px_per_m
            self.cumulative_x_m += movement_m

        self._last_athlete_x_px = athlete_px
        self._last_frame_idx = frame_idx

        # Lateral position (y) = which lane the athlete is in
        # Find nearest lane line and compute offset from it
        nearest_line_x = min(line_positions, key=lambda x: abs(x - athlete_px))
        lateral_offset_px = athlete_px - nearest_line_x
        lateral_offset_m = lateral_offset_px / smooth_px_per_m
        # Approximate lane position
        world_y = (self.calibration.runner_lane - 1) * 1.22 + lateral_offset_m

        return LaneReading(
            frame_idx=frame_idx,
            timestamp=timestamp,
            world_x=max(0.0, self.cumulative_x_m),
            world_y=world_y,
            px_per_m=smooth_px_per_m,
            lines_found=len(self.lane_lines)
        )

    def debug_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw detected lane lines and scale info on frame for debugging."""
        out = frame.copy()
        h = frame.shape[0]

        for line in self.lane_lines:
            x = int(line.x_pixel)
            cv2.line(out, (x, 0), (x, h), (0, 255, 0), 2)

        if self.px_per_m:
            cv2.putText(out, f"Scale: {self.px_per_m:.1f} px/m",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(out, f"Lines: {len(self.lane_lines)}  x={self.cumulative_x_m:.1f}m",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return out


def pick_start_line_interactive(frame: np.ndarray) -> Optional[float]:
    """
    Simple one-click calibration: click anywhere on the start line.
    Returns the x pixel coordinate clicked, or None if cancelled.
    """
    result = [None]
    display = frame.copy()
    cv2.putText(display, "Click anywhere on the START LINE, then press Q",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            result[0] = float(x)
            cv2.line(display, (x, 0), (x, frame.shape[0]), (0, 255, 255), 2)
            cv2.putText(display, f"Start line at x={x}px — press Q to confirm",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)

    while True:
        cv2.imshow("Calibration", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return result[0]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python lane_tracker.py <video_path> [lane_number]")
        sys.exit(1)

    video_path = sys.argv[1]
    runner_lane = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Could not read video")
        sys.exit(1)

    # Quick test: show what lane detection finds on frame 1
    lines = detect_lane_lines(first_frame)
    print(f"Detected {len(lines)} lane lines at x positions: {[f'{x:.0f}' for x in lines]}")

    scale = find_lane_scale(lines)
    if scale:
        print(f"Estimated scale: {scale:.1f} px/m (lane width = {scale*1.22:.0f}px)")
    else:
        print("Could not estimate scale — need at least 2 lane lines")

    # Show debug view
    debug = first_frame.copy()
    h = first_frame.shape[0]
    for x in lines:
        cv2.line(debug, (int(x), 0), (int(x), h), (0, 255, 0), 2)

    cv2.imshow("Lane detection test", debug)
    print("Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()