import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional


@dataclass
class CalibrationPoints:
    """
    Pairs of (pixel_x, pixel_y) -> (real_world_x_meters, real_world_y_meters)
    Minimum 4 points needed, more = better accuracy.

    For a sprint track, good reference points are:
    - Lane line intersections with distance markers (0m, 10m, 20m... start/finish)
    - Start blocks position
    - Finish line
    Real world coords: x = distance down track, y = lateral lane position
    """
    pixel_points: list       # [(px, py), ...]
    real_world_points: list  # [(x_m, y_m), ...]


class HomographyTransformer:
    def __init__(self, calibration: CalibrationPoints):
        if len(calibration.pixel_points) < 4:
            raise ValueError("Need at least 4 calibration points for homography")
        if len(calibration.pixel_points) != len(calibration.real_world_points):
            raise ValueError("Pixel points and real world points must match in count")

        src = np.array(calibration.pixel_points, dtype=np.float32)
        dst = np.array(calibration.real_world_points, dtype=np.float32)

        self.matrix, status = cv2.findHomography(src, dst, cv2.RANSAC)
        if self.matrix is None:
            raise ValueError("Could not compute homography — check your calibration points")

        self.calibration = calibration
        self._validate(src, dst)

    def _validate(self, src, dst):
        """Reproject calibration points through the matrix and check error"""
        errors = []
        for i in range(len(src)):
            real = self.pixel_to_world(src[i][0], src[i][1])
            expected = dst[i]
            error = np.sqrt((real[0] - expected[0])**2 + (real[1] - expected[1])**2)
            errors.append(error)

        self.mean_error_m = float(np.mean(errors))
        self.max_error_m = float(np.max(errors))

    def pixel_to_world(self, px: float, py: float) -> tuple:
        """Convert a pixel coordinate to real-world metres"""
        point = np.array([[[px, py]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.matrix)
        x, y = transformed[0][0]
        return float(x), float(y)

    def calibration_report(self) -> str:
        lines = [
            "Homography Calibration Report",
            f"  Points used: {len(self.calibration.pixel_points)}",
            f"  Mean reprojection error: {self.mean_error_m:.3f}m",
            f"  Max reprojection error: {self.max_error_m:.3f}m",
        ]
        if self.mean_error_m > 0.5:
            lines.append("  WARNING: error >0.5m — recalibrate with better/more reference points")
        elif self.mean_error_m > 0.15:
            lines.append("  CAUTION: error >0.15m — speed numbers may drift slightly")
        else:
            lines.append("  GOOD: calibration looks accurate")
        return "\n".join(lines)


class CameraMotionTracker:
    """
    Tracks camera motion (pan/zoom/drift) across frames by following static
    background features — track lines, lane markers, anything that doesn't
    move with the runners — and accumulates a homography that maps the
    CURRENT frame's pixels back into calibration-frame pixel space.

    Use this BEFORE calling HomographyTransformer.pixel_to_world() whenever
    the camera isn't locked off. Calibrate once on frame 0, then correct
    every later frame back to that reference before converting to world coords.
    """
    def __init__(self, reference_frame: np.ndarray, max_features: int = 200):
        self.reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        self.max_features = max_features
        self.prev_gray = self.reference_gray.copy()
        self.prev_points = self._detect_features(self.prev_gray)
        # Maps CURRENT frame pixel -> calibration (reference) frame pixel
        self.cumulative_H = np.eye(3, dtype=np.float64)
        self.frames_since_reseed = 0
        self.reseed_interval = 30  # features drift out of frame, refresh periodically
        self.last_inlier_ratio = 1.0  # exposed so caller can flag bad frames

    def _detect_features(self, gray):
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_features, qualityLevel=0.01, minDistance=10
        )

    def update(self, frame: np.ndarray) -> np.ndarray:
        """Call once per frame, in frame order. Returns cumulative_H."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_points is None or len(self.prev_points) < 10:
            self.prev_points = self._detect_features(self.prev_gray)
            if self.prev_points is None:
                self.prev_gray = gray
                return self.cumulative_H

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_points, None
        )
        status = status.reshape(-1)
        good_prev = self.prev_points[status == 1]
        good_next = next_points[status == 1]

        if len(good_prev) >= 8:
            H_step, inlier_mask = cv2.findHomography(good_next, good_prev, cv2.RANSAC, 3.0)
            if H_step is not None:
                self.cumulative_H = H_step @ self.cumulative_H
                if inlier_mask is not None and len(inlier_mask) > 0:
                    self.last_inlier_ratio = float(inlier_mask.sum()) / len(inlier_mask)

        self.frames_since_reseed += 1
        if self.frames_since_reseed >= self.reseed_interval or len(good_next) < 30:
            self.prev_points = self._detect_features(gray)
            self.frames_since_reseed = 0
        else:
            self.prev_points = good_next.reshape(-1, 1, 2)

        self.prev_gray = gray
        return self.cumulative_H

    def map_to_reference(self, px: float, py: float) -> tuple:
        """Map a pixel from the CURRENT frame back to calibration-frame pixel space."""
        point = np.array([[[px, py]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.cumulative_H.astype(np.float32))
        return float(mapped[0][0][0]), float(mapped[0][0][1])


def build_track_calibration_template() -> str:
    """
    Returns instructions for how to build calibration points for a running track.
    Print this so the user knows what pixel coordinates to click/measure.
    """
    return """
HOW TO CALIBRATE FOR A SPRINT TRACK:

1. Pick a frame where the track markings are clearly visible (start line, 
   finish line, or distance markers like every 10m).

2. For each marking you can identify, note:
   - The PIXEL coordinate where it appears in the frame (px, py)
   - The REAL WORLD coordinate in metres (x = distance along track, 
     y = lateral lane position, e.g. lane 1 center = 1.22m, lane 2 = 2.44m etc.)

3. You need at least 4 points, but 6-8 spread across the track gives much 
   better accuracy (especially if the camera has any angle/perspective).

EXAMPLE for a 100m race filmed from the side with markers every 20m:
   pixel_points = [
       (120, 980),   # start line, near lane edge
       (640, 960),   # 20m mark
       (1150, 940),  # 40m mark
       (1680, 920),  # 60m mark
       (2200, 900),  # 80m mark
       (2720, 880),  # finish line
   ]
   real_world_points = [
       (0, 0),
       (20, 0),
       (40, 0),
       (60, 0),
       (80, 0),
       (100, 0),
   ]

TIP: Use cv2.imshow with mouse click callback to manually click points on 
a paused frame and print their pixel coords — much easier than guessing.
"""


def pick_calibration_points_interactive(frame: np.ndarray, num_points: int = 6) -> list:
    """
    Opens a window where you click points on the frame.
    Returns list of (px, py) in the order you clicked.
    Press 'q' when done if you click fewer than num_points.
    """
    points = []
    display = frame.copy()

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < num_points:
            points.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(display, str(len(points)), (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.namedWindow("Click calibration points (q to finish)")
    cv2.setMouseCallback("Click calibration points (q to finish)", on_click)

    while True:
        cv2.imshow("Click calibration points (q to finish)", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(points) >= num_points:
            break

    cv2.destroyAllWindows()
    return points


if __name__ == "__main__":
    print(build_track_calibration_template())

    # Sanity test — points need lateral spread (not a perfectly straight line)
    # to give a numerically stable homography. Real lane lines work fine
    # because lanes have width (y varies too), this fake data just needs jitter.
    test_calibration = CalibrationPoints(
        pixel_points=[
            (120, 980), (120, 1040),     # start line, two lane edges
            (1150, 940), (1150, 1000),   # 40m mark, two lane edges
            (2720, 880), (2720, 940)     # finish line, two lane edges
        ],
        real_world_points=[
            (0, 0), (0, 1.22),
            (40, 0), (40, 1.22),
            (100, 0), (100, 1.22)
        ]
    )

    transformer = HomographyTransformer(test_calibration)
    print(transformer.calibration_report())

    test_px, test_py = 900, 950
    world_x, world_y = transformer.pixel_to_world(test_px, test_py)
    print(f"\nPixel ({test_px}, {test_py}) -> World ({world_x:.2f}m, {world_y:.2f}m)")