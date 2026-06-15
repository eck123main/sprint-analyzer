import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Optional

@dataclass
class Detection:
    track_id: int
    bbox: tuple          # (x1, y1, x2, y2) in pixels
    confidence: float
    center_x: float
    center_y: float

def load_model(model_size: str = "yolov8n") -> YOLO:
    """
    model_size options: yolov8n (fast), yolov8s, yolov8m, yolov8l, yolov8x (accurate)
    First run will auto-download the weights (~6MB for nano)
    """
    return YOLO(f"{model_size}.pt")

def detect_people(model: YOLO, frame: np.ndarray, confidence: float = 0.4) -> list[Detection]:
    """
    Runs YOLO on a single frame, returns list of people detected.
    Uses YOLO's built-in tracker so IDs are consistent across frames.
    """
    results = model.track(frame, persist=True, classes=[0], conf=confidence, verbose=False)

    detections = []
    if results[0].boxes is None:
        return detections

    boxes = results[0].boxes
    for box in boxes:
        # Skip if no track ID assigned yet
        if box.id is None:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        track_id = int(box.id[0])
        conf = float(box.conf[0])
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        detections.append(Detection(
            track_id=track_id,
            bbox=(x1, y1, x2, y2),
            confidence=conf,
            center_x=cx,
            center_y=cy
        ))

    return detections

def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes and IDs on frame for debugging."""
    frame = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID:{d.track_id} {d.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, (int(d.center_x), int(d.center_y)), 4, (0, 0, 255), -1)
    return frame

if __name__ == "__main__":
    import os
    from video_processor import extract_frames, get_video_metadata

    sample_dir = os.path.join(os.path.dirname(__file__), "../../data/sample_videos")
    videos = [f for f in os.listdir(sample_dir) if f.endswith((".mp4", ".mov", ".avi"))]

    if not videos:
        print("No video found in data/sample_videos/")
    else:
        video_path = os.path.join(sample_dir, videos[0])
        print(f"Loading model...")
        model = load_model("yolov8n")

        print(f"Running detection on: {videos[0]}")
        for frame_idx, timestamp, frame in extract_frames(video_path, skip_frames=3):
            detections = detect_people(model, frame)
            annotated = draw_detections(frame, detections)

            # Show live preview
            cv2.imshow("Detection", annotated)
            print(f"Frame {frame_idx} @ {timestamp:.2f}s — {len(detections)} people: {[d.track_id for d in detections]}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()