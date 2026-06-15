import cv2
import os
from dataclasses import dataclass
from typing import Generator

@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float

def get_video_metadata(video_path: str) -> VideoMetadata:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()
    return VideoMetadata(fps, frame_count, width, height, duration)

def extract_frames(video_path: str, skip_frames: int = 1) -> Generator:
    """
    Yields (frame_index, timestamp_seconds, frame) for each frame.
    skip_frames=1 means every frame, 2 means every other frame etc.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % skip_frames == 0:
            timestamp = frame_index / fps
            yield frame_index, timestamp, frame

        frame_index += 1

    cap.release()

def test_video_processor(video_path: str):
    print(f"Testing with: {video_path}")
    meta = get_video_metadata(video_path)
    print(f"  FPS: {meta.fps}")
    print(f"  Frames: {meta.frame_count}")
    print(f"  Size: {meta.width}x{meta.height}")
    print(f"  Duration: {meta.duration_seconds:.2f}s")

    for i, (idx, ts, frame) in enumerate(extract_frames(video_path)):
        print(f"  Frame {idx} at {ts:.3f}s — shape: {frame.shape}")
        if i >= 4:  # just show first 5
            print("  ...")
            break

if __name__ == "__main__":
    # Drop any video file into data/sample_videos/ and test it
    sample_dir = os.path.join(os.path.dirname(__file__), "../../data/sample_videos")
    videos = [f for f in os.listdir(sample_dir) if f.endswith((".mp4", ".mov", ".avi"))]

    if not videos:
        print("No video found in data/sample_videos/ — add one to test")
    else:
        test_video_processor(os.path.join(sample_dir, videos[0]))