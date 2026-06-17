import os
import sys
import cv2
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../core"))

from video_processor import extract_frames, get_video_metadata
from detector import load_model, detect_people, Detection
from tracker import AthleteTracker
from pose_estimator import load_pose_model, get_hip_position
from homography import HomographyTransformer, CalibrationPoints, pick_calibration_points_interactive
from speed_calculator import calculate_speed_profile, predict_winner, compare_athletes_at_time


CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "../../data/calibration/last_calibration.json")


def run_calibration_step(video_path: str) -> HomographyTransformer:
    """
    Pauses on the first frame, lets you click calibration points,
    then asks for real-world distances for each point.
    Saves calibration to disk so you don't have to redo it every run.
    """
    if os.path.exists(CALIBRATION_FILE):
        use_saved = input("Saved calibration found. Use it? (y/n): ").strip().lower()
        if use_saved == "y":
            with open(CALIBRATION_FILE, "r") as f:
                data = json.load(f)
            calib = CalibrationPoints(
                pixel_points=[tuple(p) for p in data["pixel_points"]],
                real_world_points=[tuple(p) for p in data["real_world_points"]]
            )
            transformer = HomographyTransformer(calib)
            print(transformer.calibration_report())
            return transformer

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read first frame for calibration")

    print("\nClick on points where you know the real-world distance (e.g. start line, "
          "10m marks, finish line). Click at least 4 points, 6+ recommended.")
    print("Press 'q' when finished clicking.\n")

    pixel_points = pick_calibration_points_interactive(frame, num_points=8)

    if len(pixel_points) < 4:
        raise ValueError(f"Only got {len(pixel_points)} points, need at least 4")

    print(f"\nGot {len(pixel_points)} points. Now enter real-world coordinates for each.")
    print("Format: distance_along_track_in_metres, lateral_lane_position_in_metres")
    print("Example: '20, 0' means 20m down the track, lane edge (y=0)\n")

    real_world_points = []
    for i, (px, py) in enumerate(pixel_points):
        while True:
            raw = input(f"Point {i+1} at pixel ({px},{py}) -> real world (x,y) in metres: ").strip()
            try:
                x_str, y_str = raw.split(",")
                real_world_points.append((float(x_str), float(y_str)))
                break
            except Exception:
                print("  Invalid format, use: x,y  e.g. 20,0")

    calib = CalibrationPoints(pixel_points=pixel_points, real_world_points=real_world_points)
    transformer = HomographyTransformer(calib)
    print(transformer.calibration_report())

    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump({
            "pixel_points": pixel_points,
            "real_world_points": real_world_points
        }, f, indent=2)
    print(f"Calibration saved to {CALIBRATION_FILE}\n")

    return transformer


def draw_overlay(frame, athletes_data, frame_idx, timestamp):
    """Draw bounding boxes, hip points, and live speed readouts on frame"""
    frame = frame.copy()
    y_offset = 40

    for track_id, info in athletes_data.items():
        bbox = info.get("bbox")
        hip = info.get("hip")
        speed = info.get("speed_mps", 0.0)
        distance = info.get("distance_m", 0.0)

        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if hip:
            cv2.circle(frame, (int(hip[0]), int(hip[1])), 6, (255, 0, 255), -1)

        label = f"ID {track_id}: {speed:.2f} m/s | {distance:.1f}m"
        text_x = int(bbox[0]) if bbox else 20
        text_y = int(bbox[1]) - 15 if bbox else y_offset
        cv2.putText(frame, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y_offset += 30

    cv2.putText(frame, f"t={timestamp:.2f}s frame={frame_idx}", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame


def run_pipeline(video_path: str, skip_frames: int = 1, model_size: str = "yolov8n"):
    print(f"Loading models...")
    yolo_model = load_model(model_size)
    pose_model = load_pose_model()
    tracker = AthleteTracker(max_missing_frames=10)

    transformer = run_calibration_step(video_path)

    # Stores raw world-position history per athlete: {track_id: [(frame, ts, wx, wy), ...]}
    world_history = {}
    # Latest per-frame display info: {track_id: {bbox, hip, speed_mps, distance_m}}
    latest_display = {}

    meta = get_video_metadata(video_path)
    print(f"\nProcessing {meta.frame_count} frames @ {meta.fps}fps...\n")

    for frame_idx, timestamp, frame in extract_frames(video_path, skip_frames=skip_frames):
        detections = detect_people(yolo_model, frame)
        athletes = tracker.update(detections, frame_idx, timestamp)

        for det in detections:
            tid = det.track_id
            keypoints = get_hip_position(pose_model, frame, det.bbox)

            if keypoints is None or keypoints.visibility < 0.4:
                # Low confidence — fall back to bbox centre, mark it as such
                hip_x, hip_y = det.center_x, det.center_y
            else:
                hip_x, hip_y = keypoints.hip_x, keypoints.hip_y

            world_x, world_y = transformer.pixel_to_world(hip_x, hip_y)

            if tid not in world_history:
                world_history[tid] = []
            world_history[tid].append((frame_idx, timestamp, world_x, world_y))

            # Quick instantaneous speed from last 2 points for live display
            speed_mps = 0.0
            hist = world_history[tid]
            if len(hist) >= 2:
                p1, p2 = hist[-2], hist[-1]
                dt = p2[1] - p1[1]
                if dt > 0:
                    dist = np.sqrt((p2[2]-p1[2])**2 + (p2[3]-p1[3])**2)
                    speed_mps = dist / dt

            latest_display[tid] = {
                "bbox": det.bbox,
                "hip": (hip_x, hip_y),
                "speed_mps": speed_mps,
                "distance_m": world_x
            }

        annotated = draw_overlay(frame, latest_display, frame_idx, timestamp)
        cv2.imshow("Sprint Analyzer", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    pose_model.close()

    print("\n=== Final Analysis ===\n")
    profiles = []
    for tid, history in world_history.items():
        profile = calculate_speed_profile(tid, history)
        profiles.append(profile)
        print(f"Athlete {tid}:")
        print(f"  Peak speed: {profile.peak_speed_mps:.2f} m/s ({profile.peak_speed_kmh:.1f} km/h)")
        print(f"  Average speed: {profile.average_speed_mps:.2f} m/s")
        print(f"  Frames tracked: {len(profile.data_points)}")
        print()

    if profiles:
        prediction = predict_winner(profiles, race_distance_m=100.0)
        print("Winner prediction:", prediction)


if __name__ == "__main__":
    sample_dir = os.path.join(os.path.dirname(__file__), "../../data/sample_videos")
    videos = [f for f in os.listdir(sample_dir) if f.endswith((".mp4", ".mov", ".avi"))]

    if not videos:
        print("No video found in data/sample_videos/")
    else:
        video_path = os.path.join(sample_dir, videos[0])
        run_pipeline(video_path, skip_frames=2)