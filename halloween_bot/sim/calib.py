"""lerobot normalized joint units <-> MuJoCo joint radians for the bimanual SO101.

lerobot (use_degrees=False) reports body joints as -100..100 across each motor's calibrated
[range_min, range_max] ticks and the gripper as 0..100. The Menagerie model (TheRobotStudio
so101_new_calib) zeroes every body joint at the middle of its range, which is the same zero
lerobot's DEGREES mode uses (mid = (min+max)/2) and the one lerobot's SO101 kinematics feed
straight into so101_new_calib. So a body joint is linear with no offset:
    rad = n/100 * (max-min)/2 ticks * 2π/4095
The gripper's 0..100 (closed..open) maps linearly onto the model's jaw range.
"""
import json
import math
from pathlib import Path

CALIB_DIR = Path(__file__).resolve().parent / "calibration"
SIDES = ("left", "right")
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
KEYS = [f"{s}_{m}.pos" for s in SIDES for m in MOTORS]  # lerobot bi_so_follower action order
TICKS_PER_RAD = 4095 / (2 * math.pi)
GRIPPER_RANGE = (-0.174533, 1.7453292)  # Menagerie jaw joint range: closed, open


def split_key(key: str) -> tuple[str, str]:
    side, motor = key.removesuffix(".pos").split("_", 1)
    return side, motor


def joint_name(key: str) -> str:
    return key.removesuffix(".pos")


def clamp_norm(key: str, value: float) -> float:
    lo, hi = (0.0, 100.0) if key.endswith("gripper.pos") else (-100.0, 100.0)
    return max(lo, min(hi, float(value)))


class Calibration:
    def __init__(self, calib: dict[str, dict], gripper_range: tuple[float, float] = GRIPPER_RANGE):
        for side in SIDES:
            for motor in MOTORS:
                if calib[side][motor].get("drive_mode", 0):
                    raise ValueError(f"{side}/{motor}: drive_mode=1 calibrations are not supported")
        self.calib = calib
        self._affine = {}
        for key in KEYS:
            side, motor = split_key(key)
            if motor == "gripper":
                lo, hi = gripper_range
                self._affine[key] = ((hi - lo) / 100.0, lo)
            else:
                c = calib[side][motor]
                half_span_rad = (c["range_max"] - c["range_min"]) / 2 / TICKS_PER_RAD
                self._affine[key] = (half_span_rad / 100.0, 0.0)

    @classmethod
    def load(cls, directory: Path = CALIB_DIR) -> "Calibration":
        return cls({s: json.loads((directory / f"bimanual_{s}.json").read_text()) for s in SIDES})

    def affine(self, key: str) -> tuple[float, float]:
        """(a, b) with rad = a * normalized + b."""
        return self._affine[key]

    def to_rad(self, key: str, n: float) -> float:
        a, b = self._affine[key]
        return a * n + b

    def to_norm(self, key: str, rad: float) -> float:
        a, b = self._affine[key]
        return (rad - b) / a
