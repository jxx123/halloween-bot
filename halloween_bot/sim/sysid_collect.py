"""Collect sys-ID traces from ONE real SO101 arm through the robot server's POST /trajectory.

Runs on the robot machine (the 1080, lerobot conda env): stdlib only, no project imports.
Only run it after an explicit go from Jinyu and a visual check that the workspace around the
chosen arm is clear.

    python sysid_collect.py --arm right --dry-run             # print the plan, touch nothing
    python sysid_collect.py --arm right --out traces_right.json

What it does, per arm:
  1. GET /state; refuse if teleop is on or any joint of the arm is > APPROACH_MAX units from
     the base pose (move it closer by hand/ctl.py first).
  2. POST /move to the base pose (SURVEY, a normal smooth move over APPROACH_SECONDS).
  3. One POST /trajectory per motor (30 Hz points, only this arm's keys, so the server keeps the
     other arm on its last command):
       body joints: base, then steps +10 / -10 / +20 / -20 held STEP_HOLD s with a return to base
                    after each (no jump ever exceeds 20 units), then a tapered linear chirp
                    (amplitude 8, 0.3 -> 2.0 Hz over 8 s), then base;
       gripper:     30 -> 50 -> 30 -> 10 -> 30 in free air (the plan's 60 is capped to base + 20).
     Every point stays within ±MAX_DEV units of the base pose and RANGE_MARGIN inside the calibrated
     range (so the elbow's +20 step from 80 stops at 95); this is checked before anything is sent.
  4. POST /move back to the starting pose.
Output: {"arm", "base", "sequences": [{"name", "hz", "trace": [{t, present, cmd}, ...]}]} which
halloween_bot.sim.sysid.load_trajectory_traces reads. Partial results are saved if a step fails.
"""
import argparse
import json
import math
import sys
import urllib.error
import urllib.request

HZ = 30
MAX_DEV = 20.0  # never command further than this from the base pose (also lerobot's max_relative_target)
RANGE_MARGIN = 5.0  # stay this far inside the calibrated range ends (mechanical stops; a stall trips overload)
MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# pick_toys.py SURVEY (+ roll/grip): a folded mid pose, far from the lift extremes
SURVEY = {"shoulder_pan": -22.0, "shoulder_lift": -63.0, "elbow_flex": 80.0, "wrist_flex": 40.0,
          "wrist_roll": -10.0, "gripper": 30.0}
STEPS = (10.0, -10.0, 20.0, -20.0)
STEP_HOLD = 1.5  # s per step level (and per return to base)
SETTLE = 1.0  # s at base at the start and end of each sequence
CHIRP = {"amp": 8.0, "f0": 0.3, "f1": 2.0, "seconds": 8.0, "taper": 0.5}
GRIPPER_LEVELS = (50.0, 30.0, 10.0, 30.0)
APPROACH_SECONDS = 4.0
APPROACH_MAX = 40.0


def base_pose(arm: str) -> dict[str, float]:
    return {f"{arm}_{m}.pos": v for m, v in SURVEY.items()}


def _hold(pose: dict, seconds: float) -> list[dict]:
    return [dict(pose) for _ in range(round(seconds * HZ))]


def _limits(key: str) -> tuple[float, float]:
    lo, hi = (0.0, 100.0) if key.endswith("gripper.pos") else (-100.0, 100.0)
    return lo + RANGE_MARGIN, hi - RANGE_MARGIN


def _safe(key: str, v: float) -> float:
    lo, hi = _limits(key)
    return min(max(v, lo), hi)


def _motor_sequence(arm: str, motor: str) -> dict:
    base = base_pose(arm)
    key = f"{arm}_{motor}.pos"
    pts = _hold(base, SETTLE)
    if motor == "gripper":
        for v in GRIPPER_LEVELS:
            pts += _hold({**base, key: _safe(key, v)}, STEP_HOLD)
    else:
        for d in STEPS:
            pts += _hold({**base, key: _safe(key, base[key] + d)}, STEP_HOLD)
            pts += _hold(base, STEP_HOLD)
        T, f0, f1 = CHIRP["seconds"], CHIRP["f0"], CHIRP["f1"]
        for i in range(round(T * HZ)):
            t = i / HZ
            env = min(1.0, t / CHIRP["taper"], (T - t) / CHIRP["taper"])
            phase = 2 * math.pi * (f0 * t + (f1 - f0) * t * t / (2 * T))
            pts.append({**base, key: _safe(key, round(base[key] + CHIRP["amp"] * env * math.sin(phase), 3))})
    pts += _hold(base, SETTLE)
    return {"name": f"{arm}_{motor}", "hz": HZ, "points": pts}


def check_sequence(seq: dict, base: dict) -> None:
    for i, p in enumerate(seq["points"]):
        if set(p) != set(base):
            raise ValueError(f"{seq['name']}[{i}]: keys {sorted(p)} are not exactly the arm's keys")
        for k, v in p.items():
            lo, hi = _limits(k)
            if abs(v - base[k]) > MAX_DEV + 1e-9 or not lo <= v <= hi:
                raise ValueError(f"{seq['name']}[{i}]: {k}={v} is outside base ± {MAX_DEV} or {lo:g}..{hi:g}")


def build_sequences(arm: str, motors=MOTORS) -> list[dict]:
    if arm not in ("left", "right"):
        raise ValueError(f"arm must be left or right, not {arm!r}")
    base = base_pose(arm)
    seqs = [_motor_sequence(arm, m) for m in motors]
    for s in seqs:
        check_sequence(s, base)
    return seqs


def describe(seqs: list[dict]) -> str:
    lines, total = [], 0.0
    for s in seqs:
        dur = len(s["points"]) / s["hz"]
        total += dur
        key = f"{s['name']}.pos"
        vals = [p[key] for p in s["points"]]
        lines.append(f"{s['name']:24s} {len(s['points']):4d} points @ {s['hz']} Hz = {dur:5.1f} s   "
                     f"{key}: {min(vals):7.2f} .. {max(vals):7.2f} (base {s['points'][0][key]:.1f})")
    lines.append(f"total trajectory time {total:.1f} s (+ {2 * APPROACH_SECONDS:.0f} s approach/return moves)")
    return "\n".join(lines)


def _call(server: str, path: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(server.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"}, method="GET" if data is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"{path}: HTTP {e.code}: {body}") from None


def _save(path: str, arm: str, results: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump({"arm": arm, "base": base_pose(arm), "sequences": results}, f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("left", "right"), required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8399")
    ap.add_argument("--out", default="traces.json")
    ap.add_argument("--motors", default=",".join(MOTORS), help="comma-separated subset of motors")
    ap.add_argument("--dry-run", action="store_true", help="print the sequences and duration; send nothing")
    a = ap.parse_args(argv)
    motors = [m.strip() for m in a.motors.split(",") if m.strip()]
    unknown = [m for m in motors if m not in MOTORS]
    if unknown:
        ap.error(f"unknown motors {unknown}; valid: {', '.join(MOTORS)}")
    seqs = build_sequences(a.arm, motors)
    base = base_pose(a.arm)
    print(f"arm={a.arm} base pose: " + ", ".join(f"{k}={v:g}" for k, v in base.items()))
    print(describe(seqs))
    if a.dry_run:
        print("dry run: nothing sent")
        return 0

    state = _call(a.server, "/state")
    if state.get("teleop"):
        print("teleop is on: turn it off first", file=sys.stderr)
        return 2
    start = {k: state["positions"][k] for k in base}
    far = {k: round(start[k] - base[k], 1) for k in base if abs(start[k] - base[k]) > APPROACH_MAX}
    if far:
        print(f"arm is too far from the base pose {far}; move it closer first", file=sys.stderr)
        return 2
    results = []
    try:
        print(f"moving {a.arm} arm to base pose over {APPROACH_SECONDS:g} s", flush=True)
        _call(a.server, "/move", {"targets": base, "duration": APPROACH_SECONDS}, timeout=APPROACH_SECONDS + 30)
        for s in seqs:
            print(f"playing {s['name']} ({len(s['points']) / s['hz']:.1f} s)", flush=True)
            out = _call(a.server, "/trajectory", {"points": s["points"], "hz": s["hz"]},
                        timeout=len(s["points"]) / s["hz"] + 30)
            if not out.get("ok"):
                raise RuntimeError(f"/trajectory {s['name']}: {out}")
            results.append({"name": s["name"], "hz": out.get("hz", s["hz"]), "trace": out["trace"]})
            _save(a.out, a.arm, results)
    finally:
        _save(a.out, a.arm, results)
        print(f"wrote {len(results)}/{len(seqs)} sequences to {a.out}; returning arm to its start pose", flush=True)
        _call(a.server, "/move", {"targets": start, "duration": APPROACH_SECONDS}, timeout=APPROACH_SECONDS + 30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
