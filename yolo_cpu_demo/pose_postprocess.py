"""Post-processing for YOLOv8-pose keypoints: detect waving, falling, and fall+wave.

Works purely on CPU with numpy. Designed for COCO 17-keypoint output from
ultralytics YOLOv8-pose models.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# COCO 17 keypoint indices
# ---------------------------------------------------------------------------
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

NUM_KEYPOINTS = 17

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
KEYPOINT_CONF_THRESH = 0.3
BODY_AXIS_FALL_ANGLE_DEG = 60.0
FALL_ANGLE_JUMP_DEG = 30.0
FALL_JUMP_FRAMES = 15          # ~0.5s @30fps
WAVE_ARM_STRAIGHT_ANGLE_DEG = 130.0
WAVE_FLIP_MIN_COUNT = 3
WAVE_WINDOW_SIZE = 25
TRACK_DISTANCE_THRESH = 100.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _point(p: np.ndarray) -> np.ndarray | None:
    """Return xy if valid (non-zero), else None."""
    if p is None or p.shape[0] < 2:
        return None
    return p[:2]


def midpoint(p1: np.ndarray | None, p2: np.ndarray | None) -> np.ndarray | None:
    if p1 is None or p2 is None:
        return None
    return (p1 + p2) * 0.5


def angle_from_vertical(vec: np.ndarray) -> float:
    """Angle (degrees) between *vec* and the downward vertical (0, +1)."""
    if vec is None or np.linalg.norm(vec) < 1e-6:
        return 0.0
    down = np.array([0.0, 1.0])
    cos_a = np.dot(vec, down) / np.linalg.norm(vec)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def three_point_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex *b* formed by rays b→a and b→c (degrees)."""
    ba = a - b
    bc = c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_a = np.clip(cos_a, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def _count_y_flips(values: list[float]) -> int:
    """Count the number of direction flips in a 1-D sequence (Y coords)."""
    if len(values) < 3:
        return 0
    flips = 0
    prev_dir = 0
    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        if abs(diff) < 1.0:
            continue
        cur_dir = 1 if diff > 0 else -1
        if prev_dir != 0 and cur_dir != prev_dir:
            flips += 1
        prev_dir = cur_dir
    return flips


# ---------------------------------------------------------------------------
# Per-person tracker (keypoint history buffer)
# ---------------------------------------------------------------------------

@dataclass
class PersonTracker:
    person_id: int
    history: deque = field(default_factory=lambda: deque(maxlen=WAVE_WINDOW_SIZE))

    def push(self, kps: np.ndarray, conf: np.ndarray, box: np.ndarray) -> None:
        self.history.append({
            "kps": kps.copy(),
            "conf": conf.copy(),
            "box": box.copy(),
        })

    @property
    def latest(self) -> dict | None:
        return self.history[-1] if self.history else None


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------

class PoseAnalyzer:
    """Analyse YOLOv8-pose results per frame and detect falling / waving."""

    def __init__(
        self,
        kpt_conf_thresh: float = KEYPOINT_CONF_THRESH,
        fall_angle: float = BODY_AXIS_FALL_ANGLE_DEG,
        wave_arm_angle: float = WAVE_ARM_STRAIGHT_ANGLE_DEG,
        wave_flip_min: int = WAVE_FLIP_MIN_COUNT,
        track_dist: float = TRACK_DISTANCE_THRESH,
    ) -> None:
        self.kpt_conf_thresh = kpt_conf_thresh
        self.fall_angle = fall_angle
        self.wave_arm_angle = wave_arm_angle
        self.wave_flip_min = wave_flip_min
        self.track_dist = track_dist
        self.trackers: dict[int, PersonTracker] = {}
        self._next_id = 0

    # -- ID association ------------------------------------------------------

    def _assign_ids(self, boxes: np.ndarray, kps: np.ndarray, conf: np.ndarray) -> dict[int, int]:
        """Match current detections to existing trackers (nearest centre).

        Returns  {detection_idx: person_id}
        """
        n = boxes.shape[0]
        if n == 0:
            return {}

        centres = np.stack([
            boxes[:, 0],  # cx from xywh
            boxes[:, 1],  # cy from xywh
        ], axis=1)

        assignment: dict[int, int] = {}
        used_trackers: set[int] = set()

        existing_centres: dict[int, np.ndarray] = {}
        for pid, trk in self.trackers.items():
            snap = trk.latest
            if snap is not None:
                existing_centres[pid] = snap["box"][:2]

        for di in range(n):
            best_pid = -1
            best_dist = self.track_dist + 1.0
            for pid, ec in existing_centres.items():
                if pid in used_trackers:
                    continue
                d = float(np.linalg.norm(centres[di] - ec))
                if d < best_dist:
                    best_dist = d
                    best_pid = pid
            if best_pid >= 0 and best_dist <= self.track_dist:
                assignment[di] = best_pid
                used_trackers.add(best_pid)
            else:
                assignment[di] = self._next_id
                self._next_id += 1

        return assignment

    # -- Per-frame analysis -------------------------------------------------

    def analyze(
        self,
        kps_xy: np.ndarray,
        kps_conf: np.ndarray,
        boxes_xywh: np.ndarray,
    ) -> dict[int, dict]:
        """Run behaviour analysis on one frame.

        Parameters
        ----------
        kps_xy   : (N, 17, 2)  keypoint x-y coordinates
        kps_conf : (N, 17)     keypoint confidences
        boxes_xywh : (N, 4)    bounding boxes (cx, cy, w, h)

        Returns
        -------
        {person_id: {"falling": bool, "waving": bool, "fall_and_wave": bool}}
        """
        n = kps_xy.shape[0]
        if n == 0:
            return {}

        assignment = self._assign_ids(boxes_xywh, kps_xy, kps_conf)

        for di in range(n):
            pid = assignment[di]
            if pid not in self.trackers:
                self.trackers[pid] = PersonTracker(person_id=pid)
            self.trackers[pid].push(kps_xy[di], kps_conf[di], boxes_xywh[di])

        results: dict[int, dict] = {}
        for di in range(n):
            pid = assignment[di]
            trk = self.trackers[pid]
            falling = self._check_falling(trk)
            waving = self._check_waving(trk)
            results[pid] = {
                "falling": falling,
                "waving": waving,
                "fall_and_wave": falling and waving,
            }
        return results

    # -- Falling ------------------------------------------------------------

    def _check_falling(self, trk: PersonTracker) -> bool:
        snap = trk.latest
        if snap is None:
            return False

        kps = snap["kps"]
        conf = snap["conf"]
        box = snap["box"]

        l_sh = self._kpt(kps, conf, LEFT_SHOULDER)
        r_sh = self._kpt(kps, conf, RIGHT_SHOULDER)
        l_hip = self._kpt(kps, conf, LEFT_HIP)
        r_hip = self._kpt(kps, conf, RIGHT_HIP)

        mid_sh = midpoint(l_sh, r_sh)
        mid_hp = midpoint(l_hip, r_hip)

        body_angle_ok = False
        bbox_ratio_ok = False

        if mid_sh is not None and mid_hp is not None:
            vec = mid_hp - mid_sh
            angle = angle_from_vertical(vec)
            if angle > self.fall_angle:
                body_angle_ok = True

        bw, bh = box[2], box[3]
        if bh > 1e-3 and bw / bh > 1.2:
            bbox_ratio_ok = True

        # High sensitivity: either condition triggers
        if body_angle_ok or bbox_ratio_ok:
            return True

        # Also check sudden angle jump in recent history
        if len(trk.history) >= 2:
            recent_angles = []
            for snap_i in list(trk.history)[-FALL_JUMP_FRAMES:]:
                k = snap_i["kps"]
                c = snap_i["conf"]
                ls = self._kpt(k, c, LEFT_SHOULDER)
                rs = self._kpt(k, c, RIGHT_SHOULDER)
                lh = self._kpt(k, c, LEFT_HIP)
                rh = self._kpt(k, c, RIGHT_HIP)
                ms = midpoint(ls, rs)
                mh = midpoint(lh, rh)
                if ms is not None and mh is not None:
                    recent_angles.append(angle_from_vertical(mh - ms))
            if len(recent_angles) >= 2:
                delta = max(recent_angles) - min(recent_angles)
                if delta > FALL_ANGLE_JUMP_DEG and recent_angles[-1] > self.fall_angle:
                    return True

        return False

    # -- Waving -------------------------------------------------------------

    def _check_waving(self, trk: PersonTracker) -> bool:
        snap = trk.latest
        if snap is None:
            return False

        kps = snap["kps"]
        conf = snap["conf"]

        nose = self._kpt(kps, conf, NOSE)
        l_sh = self._kpt(kps, conf, LEFT_SHOULDER)
        r_sh = self._kpt(kps, conf, RIGHT_SHOULDER)
        l_el = self._kpt(kps, conf, LEFT_ELBOW)
        r_el = self._kpt(kps, conf, RIGHT_ELBOW)
        l_wr = self._kpt(kps, conf, LEFT_WRIST)
        r_wr = self._kpt(kps, conf, RIGHT_WRIST)

        wave_now = False
        flip_side: str | None = None

        # Check left arm
        if l_sh is not None and l_wr is not None and nose is not None:
            if l_wr[1] < nose[1]:
                if l_el is not None:
                    arm_angle = three_point_angle(l_sh, l_el, l_wr)
                    if arm_angle > self.wave_arm_angle:
                        wave_now = True
                        flip_side = "left"
                else:
                    wave_now = True
                    flip_side = "left"

        # Check right arm
        if not wave_now and r_sh is not None and r_wr is not None and nose is not None:
            if r_wr[1] < nose[1]:
                if r_el is not None:
                    arm_angle = three_point_angle(r_sh, r_el, r_wr)
                    if arm_angle > self.wave_arm_angle:
                        wave_now = True
                        flip_side = "right"
                else:
                    wave_now = True
                    flip_side = "right"

        if not wave_now:
            return False

        # Time-series: count Y-flips of the waving wrist
        wr_idx = LEFT_WRIST if flip_side == "left" else RIGHT_WRIST
        y_values: list[float] = []
        for snap_i in trk.history:
            c = snap_i["conf"][wr_idx]
            if c < self.kpt_conf_thresh:
                continue
            y_values.append(float(snap_i["kps"][wr_idx][1]))

        flips = _count_y_flips(y_values)
        return flips >= self.wave_flip_min

    # -- helpers ------------------------------------------------------------

    def _kpt(self, kps: np.ndarray, conf: np.ndarray, idx: int) -> np.ndarray | None:
        if conf[idx] < self.kpt_conf_thresh:
            return None
        p = kps[idx][:2]
        if np.all(p == 0):
            return None
        return p


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

_LABEL_FALL = "FALLING!"
_LABEL_WAVE = "WAVING FOR HELP!"
_LABEL_BOTH = "FALL + WAVE!"

_COLOR_FALL = (0, 0, 255)       # red
_COLOR_WAVE = (0, 255, 255)     # yellow
_COLOR_BOTH = (0, 140, 255)    # orange

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 1.0
_FONT_THICK = 2


def draw_behaviour_labels(
    frame: np.ndarray,
    results: dict[int, dict],
    trackers: dict[int, PersonTracker],
) -> np.ndarray:
    """Draw behaviour labels on the frame next to each person."""
    for pid, info in results.items():
        trk = trackers.get(pid)
        if trk is None or trk.latest is None:
            continue
        box = trk.latest["box"]
        cx, cy, bw, bh = box[:4]
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2) - 10

        falling = info["falling"]
        waving = info["waving"]
        fall_and_wave = info["fall_and_wave"]

        if fall_and_wave:
            label = _LABEL_BOTH
            color = _COLOR_BOTH
        elif falling:
            label = _LABEL_FALL
            color = _COLOR_FALL
        elif waving:
            label = _LABEL_WAVE
            color = _COLOR_WAVE
        else:
            continue

        (tw, th), _ = cv2.getTextSize(label, _FONT, _FONT_SCALE, _FONT_THICK)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), _FONT, _FONT_SCALE, (255, 255, 255), _FONT_THICK, cv2.LINE_AA)

    return frame
