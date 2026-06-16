import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional

mp_pose = mp.solutions.pose

@dataclass
class PoseKeypoints:
    hip_x: float
    hip_y: float
    visibility: float
    all_landmarks: object  # raw mediapipe landmarks if needed later

def load_pose_model():
    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

def get_hip_position(pose_model, frame: np.ndarray, bbox: tuple = None) -> Optional[PoseKeypoints]:
    """
    Runs pose estimation on a frame (or cropped region if bbox given).
    Returns hip centre position — far more accurate than bbox centre
    because it's actual body centre of mass, not affected by arm swing etc.
    """
    h, w = frame.shape[:2]

    # If bbox provided, crop to that region first (much faster + more accurate)
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        offset_x, offset_y = x1, y1
        crop_h, crop_w = crop.shape[:2]
    else:
        crop = frame
        offset_x, offset_y = 0, 0
        crop_h, crop_w = h, w

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    results = pose_model.process(rgb)

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

    # Average of both hips = centre of mass approximation
    hip_x_norm = (left_hip.x + right_hip.x) / 2
    hip_y_norm = (left_hip.y + right_hip.y) / 2
    visibility = (left_hip.visibility + right_hip.visibility) / 2

    # Convert from normalized (0-1) crop coords back to full frame pixel coords
    hip_x = offset_x + (hip_x_norm * crop_w)
    hip_y = offset_y + (hip_y_norm * crop_h)

    return PoseKeypoints(hip_x=hip_x, hip_y=hip_y, visibility=visibility, all_landmarks=landmarks)

def draw_pose_point(frame: np.ndarray, keypoints: PoseKeypoints) -> np.ndarray:
    frame = frame.copy()
    if keypoints:
        cv2.circle(frame, (int(keypoints.hip_x), int(keypoints.hip_y)), 8, (255, 0, 255), -1)
        cv2.putText(frame, f"hip ({keypoints.visibility:.2f})",
                    (int(keypoints.hip_x) + 10, int(keypoints.hip_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    return frame


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from video_processor import extract_frames
    from detector import load_model, detect_people, draw_detections

    sample_dir = os.path.join(os.path.dirname(__file__), "../../data/sample_videos")
    videos = [f for f in os.listdir(sample_dir) if f.endswith((".mp4", ".mov", ".avi"))]

    if not videos:
        print("No video found")
    else:
        video_path = os.path.join(sample_dir, videos[0])
        yolo_model = load_model("yolov8n")
        pose_model = load_pose_model()

        for frame_idx, timestamp, frame in extract_frames(video_path, skip_frames=3):
            detections = detect_people(yolo_model, frame)
            annotated = draw_detections(frame, detections)

            # Get pose for first detected person (we'll handle multiple later)
            if detections:
                keypoints = get_hip_position(pose_model, frame, detections[0].bbox)
                if keypoints:
                    annotated = draw_pose_point(annotated, keypoints)
                    if frame_idx % 30 == 0:
                        print(f"Frame {frame_idx} @ {timestamp:.2f}s — hip at ({keypoints.hip_x:.0f}, {keypoints.hip_y:.0f}) vis={keypoints.visibility:.2f}")

            cv2.imshow("Pose Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        pose_model.close()