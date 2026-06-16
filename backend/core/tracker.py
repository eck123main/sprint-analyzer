from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class TrackedAthlete:
    track_id: int
    center_x: float
    center_y: float
    bbox: tuple
    confidence: float
    frames_missing: int = 0
    position_history: list = field(default_factory=list)  # list of (frame_idx, ts, cx, cy)

class AthleteTracker:
    def __init__(self, max_missing_frames: int = 10):
        """
        max_missing_frames: how many frames an athlete can disappear
        before we consider them truly gone (not just occluded)
        """
        self.athletes: dict[int, TrackedAthlete] = {}
        self.max_missing_frames = max_missing_frames

    def update(self, detections, frame_idx: int, timestamp: float) -> dict[int, TrackedAthlete]:
        """
        Feed in detections from detector.py for this frame.
        Returns current active athletes with stable IDs.
        """
        seen_ids = set()

        for det in detections:
            tid = det.track_id
            seen_ids.add(tid)

            if tid not in self.athletes:
                # New athlete
                self.athletes[tid] = TrackedAthlete(
                    track_id=tid,
                    center_x=det.center_x,
                    center_y=det.center_y,
                    bbox=det.bbox,
                    confidence=det.confidence
                )

            # Update position
            athlete = self.athletes[tid]
            athlete.center_x = det.center_x
            athlete.center_y = det.center_y
            athlete.bbox = det.bbox
            athlete.confidence = det.confidence
            athlete.frames_missing = 0
            athlete.position_history.append((frame_idx, timestamp, det.center_x, det.center_y))

        # Increment missing counter for athletes not seen this frame
        to_remove = []
        for tid, athlete in self.athletes.items():
            if tid not in seen_ids:
                athlete.frames_missing += 1
                if athlete.frames_missing > self.max_missing_frames:
                    to_remove.append(tid)

        for tid in to_remove:
            del self.athletes[tid]

        return self.athletes

    def get_active_athletes(self) -> list[TrackedAthlete]:
        return list(self.athletes.values())

    def summary(self) -> str:
        lines = [f"Tracking {len(self.athletes)} athletes:"]
        for tid, a in self.athletes.items():
            lines.append(f"  ID {tid}: pos=({a.center_x:.0f}, {a.center_y:.0f}) missing={a.frames_missing} history={len(a.position_history)} frames")
        return "\n".join(lines)


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from video_processor import extract_frames
    from detector import load_model, detect_people, draw_detections
    import cv2

    sample_dir = os.path.join(os.path.dirname(__file__), "../../data/sample_videos")
    videos = [f for f in os.listdir(sample_dir) if f.endswith((".mp4", ".mov", ".avi"))]

    if not videos:
        print("No video found")
    else:
        video_path = os.path.join(sample_dir, videos[0])
        model = load_model("yolov8n")
        tracker = AthleteTracker(max_missing_frames=10)

        for frame_idx, timestamp, frame in extract_frames(video_path, skip_frames=3):
            detections = detect_people(model, frame)
            athletes = tracker.update(detections, frame_idx, timestamp)

            annotated = draw_detections(frame, detections)
            cv2.imshow("Tracker", annotated)

            if frame_idx % 30 == 0:
                print(f"\n--- Frame {frame_idx} @ {timestamp:.2f}s ---")
                print(tracker.summary())

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        print("\n=== Final Summary ===")
        print(tracker.summary())
        for tid, athlete in tracker.athletes.items():
            print(f"Athlete {tid}: tracked for {len(athlete.position_history)} frames")