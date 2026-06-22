import os
import sys
import cv2
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../core"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../utils"))

from video_processor import extract_frames, get_video_metadata
from detector import load_model, detect_people, Detection
from tracker import AthleteTracker
from pose_estimator import load_pose_model, get_hip_position
from homography import (
    HomographyTransformer,
    CalibrationPoints,
    pick_calibration_points_interactive,
    CameraMotionTracker,
)
from speed_calculator import calculate_speed_profile, predict_winner, compare_athletes_at_time
from occlusion_handler import OcclusionHandler


CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "../../data/calibration/last_calibration.json")


def run_calibration_step(video_path: str):
    """
    Pauses on the first frame, lets you click calibration points,
    then asks for real-world distances for each point.
    Saves calibration to disk so you don't have to redo it every run.

    Returns (HomographyTransformer, calibration_frame). The calibration
    frame is also needed to seed CameraMotionTracker, since that's the
    reference frame all later frames get corrected back to.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read first frame for calibration")

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
            return transformer, frame

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

    return transformer, frame


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

    transformer, calibration_frame = run_calibration_step(video_path)

    # Corrects for camera pan/zoom/drift by tracking static background
    # features and mapping every later frame back to the calibration frame
    # before converting pixels to world coordinates.
    motion_tracker = CameraMotionTracker(calibration_frame)

    # Detects bbox overlaps between athletes and corrects likely ID swaps
    # (e.g. two sprinters crossing paths from a side-on camera) using lane
    # position consistency. See utils/occlusion_handler.py.
    occlusion_handler = OcclusionHandler()

    # Latest per-frame display info only: {track_id: {bbox, hip, speed_mps, distance_m}}
    # World-space position HISTORY now lives entirely on the tracker
    # (TrackedAthlete.position_history), not in a separate dict here.
    latest_display = {}

    meta = get_video_metadata(video_path)
    print(f"\nProcessing {meta.frame_count} frames @ {meta.fps}fps...\n")

    for frame_idx, timestamp, frame in extract_frames(video_path, skip_frames=skip_frames):
        # Update camera motion estimate every frame, BEFORE using it below
        motion_tracker.update(frame)
        if motion_tracker.last_inlier_ratio < 0.5:
            print(f"  WARNING frame {frame_idx}: low motion-tracking confidence "
                  f"({motion_tracker.last_inlier_ratio:.2f}) — camera correction may be unreliable")

        detections = detect_people(yolo_model, frame)
        athletes = tracker.update(detections, frame_idx, timestamp)

        # Which IDs are in (or just came out of) a bbox overlap this frame —
        # only these are eligible for ID-swap correction below.
        suspect_ids = occlusion_handler.note_occlusion_candidates(detections)

        # PASS 1: compute hip position + world coords for every detection
        # this frame, without writing anything yet — the swap-correction
        # step needs to see everyone's position in this frame at once.
        raw_world_positions = {}   # tid -> (world_x, world_y)
        per_det_info = {}          # tid -> {bbox, hip_x, hip_y}

        for det in detections:
            tid = det.track_id
            keypoints = get_hip_position(pose_model, frame, det.bbox)

            if keypoints is None or keypoints.visibility < 0.4:
                # Low confidence — fall back to bbox centre, mark it as such
                hip_x, hip_y = det.center_x, det.center_y
            else:
                hip_x, hip_y = keypoints.hip_x, keypoints.hip_y

            # Correct for camera motion BEFORE converting to world coords —
            # maps this frame's pixel back into the calibration frame's
            # pixel space, so the static homography still applies correctly
            # even if the camera panned/zoomed since calibration.
            ref_x, ref_y = motion_tracker.map_to_reference(hip_x, hip_y)
            world_x, world_y = transformer.pixel_to_world(ref_x, ref_y)

            raw_world_positions[tid] = (world_x, world_y)
            per_det_info[tid] = {"bbox": det.bbox, "hip": (hip_x, hip_y)}

        # PASS 2: resolve likely ID swaps among this frame's positions,
        # using lane consistency. id_mapping is identity for anyone not
        # under suspicion this frame.
        id_mapping = occlusion_handler.update_and_correct(
            raw_world_positions, frame_idx, timestamp, suspect_ids
        )

        # PASS 3: write world history and live-display info under the
        # CORRECTED id, so a swap doesn't graft one athlete's data onto
        # another's history.
        for original_tid, (world_x, world_y) in raw_world_positions.items():
            corrected_tid = id_mapping.get(original_tid, original_tid)
            info = per_det_info[original_tid]

            tracker.update_world_position(corrected_tid, frame_idx, timestamp, world_x, world_y)

            speed_mps = 0.0
            hist = tracker.athletes[corrected_tid].position_history
            if len(hist) >= 2:
                p1, p2 = hist[-2], hist[-1]
                dt = p2[1] - p1[1]
                if dt > 0:
                    dist = np.sqrt((p2[2] - p1[2]) ** 2 + (p2[3] - p1[3]) ** 2)
                    speed_mps = dist / dt

            latest_display[corrected_tid] = {
                "bbox": info["bbox"],
                "hip": info["hip"],
                "speed_mps": speed_mps,
                "distance_m": world_x
            }

        annotated = draw_overlay(frame, latest_display, frame_idx, timestamp)
        cv2.imshow("Sprint Analyzer", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    pose_model.close()

    print("\n=== Occlusion / ID-Swap Report ===\n")
    print(occlusion_handler.report())

    print("\n=== Final Analysis ===\n")
    profiles = []
    # all_athletes_ever_tracked() includes both still-active AND archived
    # athletes, so a brief occlusion or finishing the race and leaving
    # frame doesn't silently drop someone's data from the final report.
    for tid, athlete in tracker.all_athletes_ever_tracked().items():
        profile = calculate_speed_profile(tid, athlete.position_history)
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