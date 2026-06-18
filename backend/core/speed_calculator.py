import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.signal import savgol_filter


@dataclass
class SpeedDataPoint:
    frame_idx: int
    timestamp: float
    world_x: float
    world_y: float
    speed_mps: float = 0.0
    speed_kmh: float = 0.0
    acceleration_mps2: float = 0.0


@dataclass
class AthleteSpeedProfile:
    track_id: int
    data_points: list = field(default_factory=list)

    @property
    def peak_speed_mps(self) -> float:
        if not self.data_points:
            return 0.0
        return max(p.speed_mps for p in self.data_points)

    @property
    def peak_speed_kmh(self) -> float:
        return self.peak_speed_mps * 3.6

    @property
    def average_speed_mps(self) -> float:
        if not self.data_points:
            return 0.0
        return float(np.mean([p.speed_mps for p in self.data_points]))

    def speed_at_distance(self, target_distance_m: float) -> Optional[float]:
        """Find speed when athlete was closest to a given distance down the track"""
        if not self.data_points:
            return None
        closest = min(self.data_points, key=lambda p: abs(p.world_x - target_distance_m))
        return closest.speed_mps

    def split_time(self, distance_m: float) -> Optional[float]:
        """Find the timestamp when athlete crossed a given distance"""
        for i in range(len(self.data_points) - 1):
            p1, p2 = self.data_points[i], self.data_points[i + 1]
            if p1.world_x <= distance_m <= p2.world_x:
                # Linear interpolation between the two frames
                if p2.world_x == p1.world_x:
                    return p1.timestamp
                ratio = (distance_m - p1.world_x) / (p2.world_x - p1.world_x)
                return p1.timestamp + ratio * (p2.timestamp - p1.timestamp)
        return None


def calculate_speed_profile(
    track_id: int,
    position_history: list,   # list of (frame_idx, timestamp, world_x, world_y)
    smooth: bool = True,
    smooth_window: int = 7
) -> AthleteSpeedProfile:
    """
    Takes raw world-coordinate position history and computes speed + acceleration
    at every frame. Raw frame-to-frame speed is noisy (tracking jitter), so we
    optionally smooth it with a Savitzky-Golay filter — preserves the overall
    curve shape while removing high-frequency noise.
    """
    if len(position_history) < 2:
        return AthleteSpeedProfile(track_id=track_id, data_points=[])

    sorted_history = sorted(position_history, key=lambda x: x[0])  # sort by frame_idx

    frames = [h[0] for h in sorted_history]
    timestamps = [h[1] for h in sorted_history]
    xs = [h[2] for h in sorted_history]
    ys = [h[3] for h in sorted_history]

    # Smooth raw positions first if requested and we have enough points
    if smooth and len(xs) >= smooth_window:
        # window must be odd and <= number of points
        window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        window = min(window, len(xs) if len(xs) % 2 == 1 else len(xs) - 1)
        if window >= 3:
            xs = savgol_filter(xs, window, polyorder=2).tolist()
            ys = savgol_filter(ys, window, polyorder=2).tolist()

    data_points = []
    speeds = [0.0]  # first frame has no speed yet

    for i in range(1, len(sorted_history)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            speeds.append(speeds[-1])
            continue

        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        distance = np.sqrt(dx**2 + dy**2)
        speed = distance / dt
        speeds.append(speed)

    # Smooth the speed curve itself too — raw frame-diff speed is very jittery
    if smooth and len(speeds) >= smooth_window:
        window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        window = min(window, len(speeds) if len(speeds) % 2 == 1 else len(speeds) - 1)
        if window >= 3:
            speeds = savgol_filter(speeds, window, polyorder=2).tolist()

    # Calculate acceleration from the smoothed speed curve
    accelerations = [0.0]
    for i in range(1, len(speeds)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt <= 0:
            accelerations.append(0.0)
            continue
        accel = (speeds[i] - speeds[i - 1]) / dt
        accelerations.append(accel)

    for i in range(len(sorted_history)):
        data_points.append(SpeedDataPoint(
            frame_idx=frames[i],
            timestamp=timestamps[i],
            world_x=xs[i],
            world_y=ys[i],
            speed_mps=max(0.0, speeds[i]),  # clamp negative noise to 0
            speed_kmh=max(0.0, speeds[i]) * 3.6,
            acceleration_mps2=accelerations[i]
        ))

    return AthleteSpeedProfile(track_id=track_id, data_points=data_points)


def compare_athletes_at_time(profiles: list, timestamp: float) -> list:
    """
    Given multiple athlete speed profiles, find who's leading at a given timestamp.
    Returns list sorted by distance covered (leader first).
    """
    results = []
    for profile in profiles:
        # Find closest data point to this timestamp
        if not profile.data_points:
            continue
        closest = min(profile.data_points, key=lambda p: abs(p.timestamp - timestamp))
        results.append({
            "track_id": profile.track_id,
            "distance_m": closest.world_x,
            "speed_mps": closest.speed_mps,
            "timestamp": closest.timestamp
        })

    results.sort(key=lambda r: r["distance_m"], reverse=True)
    return results


def predict_winner(profiles: list, race_distance_m: float = 100.0) -> dict:
    """
    Simple winner prediction: extrapolate each athlete's current speed/trend
    to estimate finish time. NOT a sophisticated model — just current velocity
    + recent acceleration projected forward. Good enough for early testing.
    """
    predictions = []

    for profile in profiles:
        if not profile.data_points:
            continue

        latest = profile.data_points[-1]
        remaining_distance = race_distance_m - latest.world_x

        if remaining_distance <= 0:
            predictions.append({
                "track_id": profile.track_id,
                "status": "finished",
                "current_distance": latest.world_x,
                "current_time": latest.timestamp
            })
            continue

        # Use current speed (already smoothed) to project remaining time
        # Slight deceleration assumption for long remaining distance (fatigue)
        effective_speed = latest.speed_mps
        if effective_speed <= 0:
            effective_speed = profile.average_speed_mps or 0.1

        estimated_remaining_time = remaining_distance / effective_speed
        estimated_finish_time = latest.timestamp + estimated_remaining_time

        predictions.append({
            "track_id": profile.track_id,
            "status": "racing",
            "current_distance": round(latest.world_x, 2),
            "current_speed_mps": round(latest.speed_mps, 2),
            "estimated_finish_time": round(estimated_finish_time, 3)
        })

    racing = [p for p in predictions if p["status"] == "racing"]
    finished = [p for p in predictions if p["status"] == "finished"]

    racing.sort(key=lambda p: p["estimated_finish_time"])
    finished.sort(key=lambda p: p["current_time"])

    return {
        "predicted_order": finished + racing,
        "predicted_winner": (finished + racing)[0]["track_id"] if (finished or racing) else None
    }


if __name__ == "__main__":
    # Quick test with synthetic data simulating a sprinter accelerating then plateauing
    import random

    fake_history = []
    x = 0.0
    speed = 0.0
    for frame in range(100):
        t = frame / 30.0  # 30fps
        # Simulate acceleration phase then top speed plateau
        if t < 3.0:
            speed = min(10.5, speed + 0.4)  # accelerating
        else:
            speed = 10.5 + random.uniform(-0.3, 0.3)  # plateau with noise
        x += speed * (1/30.0)
        y = 1.22 + random.uniform(-0.05, 0.05)  # slight lane wobble
        fake_history.append((frame, t, x, y))

    profile = calculate_speed_profile(track_id=1, position_history=fake_history)

    print(f"Peak speed: {profile.peak_speed_mps:.2f} m/s ({profile.peak_speed_kmh:.1f} km/h)")
    print(f"Average speed: {profile.average_speed_mps:.2f} m/s")
    print(f"Split at 20m: {profile.split_time(20):.2f}s" if profile.split_time(20) else "20m not reached")
    print(f"Split at 50m: {profile.split_time(50):.2f}s" if profile.split_time(50) else "50m not reached")

    print("\nSpeed curve sample (every 10 frames):")
    for p in profile.data_points[::10]:
        print(f"  t={p.timestamp:.2f}s  x={p.world_x:.1f}m  speed={p.speed_mps:.2f}m/s  accel={p.acceleration_mps2:.2f}m/s2")

    print("\nWinner prediction test:")
    prediction = predict_winner([profile], race_distance_m=100.0)
    print(prediction)