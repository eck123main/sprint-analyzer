import itertools
from dataclasses import dataclass, field
from typing import Optional


def bbox_iou(box_a: tuple, box_b: tuple) -> float:
    """Standard intersection-over-union between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


@dataclass
class SwapEvent:
    frame_idx: int
    timestamp: float
    mapping: dict  # {original_tid: corrected_tid}
    reason: str


class OcclusionHandler:
    """
    Detects bbox overlaps between athletes (the only situation where YOLO's
    tracker can plausibly mismatch identities) and corrects likely ID swaps
    using lane position (world_y) as the consistency signal — sprinters stay
    in roughly the same lane for the whole race, so a sudden world_y jump
    right after an overlap is a strong tell that two IDs got crossed.

    This is a heuristic correction, not a certainty. Every correction it
    makes gets logged via swap_log so you can audit/disable specific ones
    if they look wrong, rather than trusting it silently.
    """

    def __init__(self,
                 iou_threshold: float = 0.10,
                 lane_tolerance_m: float = 0.6,
                 lane_ema_alpha: float = 0.15,
                 cooldown_frames: int = 8):
        self.iou_threshold = iou_threshold
        self.lane_tolerance_m = lane_tolerance_m
        self.lane_ema_alpha = lane_ema_alpha
        self.cooldown_frames = cooldown_frames

        self.lane_estimates: dict[int, float] = {}      # tid -> smoothed world_y
        self.cooldowns: dict[int, int] = {}              # tid -> frames remaining under suspicion
        self.swap_log: list[SwapEvent] = []
        self.active_swaps: dict = {}                      # original_tid -> corrected_tid currently in effect

    def note_occlusion_candidates(self, detections: list) -> set:
        """
        Call once per frame with this frame's Detection list.
        Returns the set of track_ids whose bboxes overlap with at least
        one other detection this frame (IOU above threshold) — these
        and their cooldown partners are the only candidates eligible
        for swap correction.
        """
        overlapping_ids = set()
        for det_a, det_b in itertools.combinations(detections, 2):
            if bbox_iou(det_a.bbox, det_b.bbox) >= self.iou_threshold:
                overlapping_ids.add(det_a.track_id)
                overlapping_ids.add(det_b.track_id)

        # Put overlapping ids (and anyone already in cooldown) into cooldown,
        # since swaps often only become visible a frame or two AFTER
        # separation, not in the exact overlap frame itself.
        for tid in overlapping_ids:
            self.cooldowns[tid] = self.cooldown_frames

        suspect_ids = set(self.cooldowns.keys())

        # Tick down cooldowns, drop expired ones
        expired = []
        for tid in self.cooldowns:
            if tid not in overlapping_ids:
                self.cooldowns[tid] -= 1
                if self.cooldowns[tid] <= 0:
                    expired.append(tid)
        for tid in expired:
            del self.cooldowns[tid]

        return suspect_ids

    def update_and_correct(self,
                            world_positions: dict,   # {tid: (world_x, world_y)}
                            frame_idx: int,
                            timestamp: float,
                            suspect_ids: set) -> dict:
        """
        Returns {original_tid: corrected_tid} for this frame. IDs not
        under suspicion map to themselves. Call this AFTER
        note_occlusion_candidates() for the same frame.
        """
        mapping = {tid: tid for tid in world_positions}

        # IDs we've never seen before just seed their lane estimate and
        # are not eligible for correction yet (nothing to compare against).
        seedable = [tid for tid in world_positions if tid not in self.lane_estimates]
        for tid in seedable:
            self.lane_estimates[tid] = world_positions[tid][1]

        # Only attempt correction among suspects that already have a
        # lane estimate AND were just reported this frame.
        candidates = [tid for tid in suspect_ids
                      if tid in world_positions and tid in self.lane_estimates
                      and tid not in seedable]

        if len(candidates) >= 2:
            # Cost of keeping current (identity) assignment
            identity_cost = sum(
                abs(world_positions[tid][1] - self.lane_estimates[tid])
                for tid in candidates
            )

            best_perm = None
            best_cost = identity_cost

            # Candidate groups are small (2-4 people overlapping at once in
            # practice), so brute-force permutation search is cheap and exact.
            for perm in itertools.permutations(candidates):
                cost = sum(
                    abs(world_positions[candidates[i]][1] - self.lane_estimates[perm[i]])
                    for i in range(len(candidates))
                )
                if cost < best_cost:
                    best_cost = cost
                    best_perm = perm

            # Only apply if swapping meaningfully beats doing nothing —
            # require improvement bigger than the lane tolerance, not just
            # any improvement, since noise alone can produce tiny deltas.
            if best_perm is not None and (identity_cost - best_cost) > self.lane_tolerance_m:
                swap_mapping = {candidates[i]: best_perm[i] for i in range(len(candidates))}
                if swap_mapping != {tid: tid for tid in candidates}:
                    for original, corrected in swap_mapping.items():
                        mapping[original] = corrected

                    # Only log/print if this is a NEW swap, not a continuation
                    # of one we already applied last frame — otherwise an
                    # ongoing swap spams one log line per frame for as long
                    # as it persists.
                    is_new = any(
                        self.active_swaps.get(original) != corrected
                        for original, corrected in swap_mapping.items()
                    )
                    if is_new:
                        self.swap_log.append(SwapEvent(
                            frame_idx=frame_idx,
                            timestamp=timestamp,
                            mapping=swap_mapping,
                            reason=f"identity_cost={identity_cost:.2f}m best_cost={best_cost:.2f}m"
                        ))
                        print(f"  [occlusion] frame {frame_idx} @ {timestamp:.2f}s: "
                              f"corrected likely ID swap {swap_mapping} "
                              f"(saved {identity_cost - best_cost:.2f}m of lane inconsistency)")
                    self.active_swaps.update(swap_mapping)
                else:
                    # Identity won this round — clear any previously active
                    # swap entries for these candidates so a future re-swap
                    # back gets logged as new, not silently ignored.
                    for tid in candidates:
                        self.active_swaps.pop(tid, None)

        # Update lane estimates using the (possibly corrected) identity
        for original_tid, corrected_tid in mapping.items():
            world_y = world_positions[original_tid][1]
            prev = self.lane_estimates.get(corrected_tid, world_y)
            self.lane_estimates[corrected_tid] = (
                (1 - self.lane_ema_alpha) * prev + self.lane_ema_alpha * world_y
            )

        return mapping

    def report(self) -> str:
        if not self.swap_log:
            return "No ID swaps detected/corrected during this run."
        lines = [f"{len(self.swap_log)} ID swap(s) corrected:"]
        for event in self.swap_log:
            lines.append(f"  frame {event.frame_idx} @ {event.timestamp:.2f}s: "
                         f"{event.mapping} ({event.reason})")
        lines.append("NOTE: these are heuristic corrections based on lane consistency. "
                     "Spot-check the source footage at these timestamps if results look off.")
        return "\n".join(lines)