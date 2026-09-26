"""lerobot normalized joint units <-> MuJoCo joint radians for the bimanual SO101.

lerobot (use_degrees=False) reports body joints as -100..100 across each motor's calibrated
[range_min, range_max] ticks and the gripper as 0..100. The Menagerie model (TheRobotStudio
so101_new_calib) zeroes every body joint at the middle of its range, which is the same zero
lerobot's DEGREES mode uses (mid = (min+max)/2) and the one lerobot's SO101 kinematics feed
straight into so101_new_calib. So a body joint is linear with no offset:
    rad = n/100 * (max-min)/2 ticks * 2π/4095
The gripper's 0..100 (closed..open) maps linearly onto the model's jaw range.
Per-joint zero convention (geometry.json "zero", from the kinematic calibration in kincal.py):
  "mid"    (default) model zero = middle of the recorded range (lerobot DEGREES mode).
  "homing" model zero = the calibration homing pose (tick 2047). lerobot calibration places the arm in
           the documented middle pose -- so101_new_calib's zero -- and writes a homing offset there;
           when a joint's mechanical stops aren't symmetric about that pose (the WOWROBO arms'
           shoulder and elbow), the range middle is NOT the model zero. Each arm uses its own file.
Plus optional fitted per-motor angle offsets (both arms).
"""
import json
import math
from pathlib import Path

CALIB_DIR = Path(__file__).resolve().parent / "calibration"
GEOMETRY_FILE = Path(__file__).resolve().parent / "geometry.json"  # kincal.py output
SIDES = ("left", "right")
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
KEYS = [f"{s}_{m}.pos" for s in SIDES for m in MOTORS]  # lerobot bi_so_follower action order
TICKS_PER_RAD = 4095 / (2 * math.pi)
GRIPPER_RANGE = (-0.174533, 1.7453292)  # Menagerie jaw joint range: closed, open


def _radians(offsets_deg: dict) -> dict:
    """{motor: deg} or {side: {motor: deg}} -> same shape in radians."""
    return {k: (_radians(v) if isinstance(v, dict) else math.radians(v)) for k, v in offsets_deg.items()}


def split_key(key: str) -> tuple[str, str]:
    side, motor = key.removesuffix(".pos").split("_", 1)
    return side, motor


def joint_name(key: str) -> str:
    return key.removesuffix(".pos")


def clamp_norm(key: str, value: float) -> float:
    lo, hi = (0.0, 100.0) if key.endswith("gripper.pos") else (-100.0, 100.0)
    return max(lo, min(hi, float(value)))


class Calibration:
    def __init__(self, calib: dict[str, dict], gripper_range: tuple[float, float] = GRIPPER_RANGE,
                 offsets: dict[str, float] | None = None, zero: dict[str, str] | None = None):
        for side in SIDES:
            for motor in MOTORS:
                if calib[side][motor].get("drive_mode", 0):
                    raise ValueError(f"{side}/{motor}: drive_mode=1 calibrations are not supported")
        self.calib = calib
        # motor -> rad for both arms, or {"left": {motor: rad}, "right": {...}} per arm
        offsets = dict(offsets or {})
        self.offsets = ({s: dict(offsets.get(s, {})) for s in SIDES} if set(offsets) & set(SIDES)
                        else {s: dict(offsets) for s in SIDES})
        self.zero = dict(zero or {})  # motor -> "mid" | "homing"
        for motor, conv in self.zero.items():
            if motor == "gripper" or motor not in MOTORS or conv not in ("mid", "homing"):
                raise ValueError(f"bad zero convention {motor}={conv}")
        self._affine = {}
        for key in KEYS:
            side, motor = split_key(key)
            if motor == "gripper":
                lo, hi = gripper_range
                self._affine[key] = ((hi - lo) / 100.0, lo)
            else:
                c = calib[side][motor]
                half_span_rad = (c["range_max"] - c["range_min"]) / 2 / TICKS_PER_RAD
                shift = 0.0
                if self.zero.get(motor) == "homing":
                    shift = ((c["range_min"] + c["range_max"]) / 2 - 2047) / TICKS_PER_RAD
                self._affine[key] = (half_span_rad / 100.0, shift + self.offsets[side].get(motor, 0.0))

    @classmethod
    def load(cls, directory: Path = CALIB_DIR, geometry: Path | None = GEOMETRY_FILE) -> "Calibration":
        offsets, zero = {}, {}
        if geometry is not None and geometry.exists():
            g = json.loads(geometry.read_text())
            offsets = _radians(g.get("joint_offsets_deg", {}))
            zero = g.get("zero", {})
        return cls({s: json.loads((directory / f"bimanual_{s}.json").read_text()) for s in SIDES},
                   offsets=offsets, zero=zero)

    def affine(self, key: str) -> tuple[float, float]:
        """(a, b) with rad = a * normalized + b."""
        return self._affine[key]

    def to_rad(self, key: str, n: float) -> float:
        a, b = self._affine[key]
        return a * n + b

    def to_norm(self, key: str, rad: float) -> float:
        a, b = self._affine[key]
        return (rad - b) / a
