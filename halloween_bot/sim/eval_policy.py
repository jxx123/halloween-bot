#!/usr/bin/env python
"""Evaluate a pi0-FAST / pi05 checkpoint on the candy tasks in the MuJoCo twin.

Each episode is set up exactly like a scripted demo (CandyExpert.setup: seeded bowl, candy, plate/hand,
instruction), then the policy runs through PolicyRunner (lockstep: physics pauses while it infers) and
is scored with the recorder's success tests. For give_human the person reaches in once the named candy
is lifted, as in the training demos.

Seeds default to 10_000+ so they don't overlap the recorded episodes (seeds < 1000).

    .venv/bin/python -m halloween_bot.sim.eval_policy --policy-server 127.0.0.1:8082 \\
        --checkpoint ~/lerobot/outputs/train/pi0fast_sim_candy_v0/merged --tasks pick_place,give_human \\
        --episodes 10 --video-dir ~/lerobot/outputs/claude_robot/eval_sim_candy_v0
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import math
import time
from pathlib import Path

import threading

import cv2
import mujoco
import numpy as np

from ..policy_runner import PI0FAST_CAMERAS, PolicyRunner
from . import candy as C
from .calib import KEYS
from .candy_expert import CandyExpert, NoCandy
from .engine import SimEngine
from .model import CAMERA_NAMES
from .server import other_policy_clients, policy_features, policy_observation


class World:
    """The scene's scripted side during a policy episode: the person's hand, and the success test."""

    def __init__(self, eng: SimEngine, task: str, ep: dict, hz: float = 30.0):
        self.eng, self.task, self.ep, self.hz = eng, task, ep, hz
        self.hand_path: list | None = None
        self.success_since: float | None = None

    def tick(self):
        m, d = self.eng.model, self.eng.data
        name = self.ep["candy"]
        if self.task == "give_human":
            lifted = d.xpos[m.body(name).id][2] > 0.06
            if self.hand_path is None and lifted:  # the person reaches in once the robot has a candy
                n = int(1.4 * self.hz)
                a, b = np.asarray(self.ep["hand_from"]), np.asarray(self.ep["hand"])
                self.hand_path = [a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * i / n)) for i in range(1, n + 1)]
            if self.hand_path:
                C.set_hand(m, d, self.hand_path.pop(0), self.ep["hand_yaw"])

    def success(self) -> bool:
        m, d, name = self.eng.model, self.eng.data, self.ep["candy"]
        return C.in_hand(m, d, name) if self.task == "give_human" else C.on_plate(m, d, name)


def smoothness(sent: list, hz: float = 30.0) -> dict:
    """How the arms were driven: idle = share of the run with no action to send (queue empty, the robot just
    holds), jumps = step-to-step command changes (units; chunk-boundary jerks show as the max / p99)."""
    if len(sent) < 3:
        return {"idle_frac": None, "jump_max": None, "jump_p99": None}
    t = np.array([s[0] for s in sent])
    a = np.stack([s[1] for s in sent])
    span = t[-1] - t[0]
    jumps = np.abs(np.diff(a, axis=0)).max(axis=1)
    return {"idle_frac": round(float(max(0.0, 1 - (len(sent) - 1) / (span * hz))), 3) if span > 0 else None,
            "jump_max": round(float(jumps.max()), 1), "jump_p99": round(float(np.percentile(jumps, 99)), 1)}


class Episodes:
    """One PolicyRunner for the whole evaluation (it instructs the server once, so the model loads once);
    the observation hook drives whichever episode is current."""

    def __init__(self, eng: SimEngine, runner_kw: dict, full_video: bool = False, pad: bool = True):
        self.eng, self.world, self.writer, self.pad = eng, None, None, pad
        self.full_video, self._tls, self.caption = full_video, threading.local(), ""
        self.sent: list[tuple[float, np.ndarray]] = []  # (wall time, action) per executed step: smoothness metrics
        self.runner = PolicyRunner(self._observe, self._send, KEYS, policy_features(pad), pause=eng.pause,
                                   resume=eng.resume, **runner_kw)

    def _send(self, action: dict):
        self.sent.append((time.perf_counter(), np.array([action[k] for k in KEYS])))
        out = self.eng.send_action(action)
        if self.full_video and self.writer is not None:  # every control step: a real-time 30 fps video
            tl = self._tls
            if not hasattr(tl, "r"):  # EGL contexts are per thread: one renderer set per runner thread
                tl.r = {c: mujoco.Renderer(self.eng.model, 360, 480) for c in ("overhead", "scene")}
            with self.eng.lock:
                imgs = []
                for cam, r in tl.r.items():
                    r.update_scene(self.eng.data, CAMERA_NAMES[cam])
                    imgs.append(r.render())
            frame = cv2.cvtColor(np.hstack(imgs), cv2.COLOR_RGB2BGR)
            cv2.putText(frame, self.caption, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            self.writer.write(frame)
        return out

    def _observe(self):
        if self.world is not None:
            self.world.tick()
        obs = policy_observation(self.eng, pad=self.pad)
        if self.writer is not None and not self.full_video:
            self.writer.write(cv2.cvtColor(cv2.resize(obs[PI0FAST_CAMERAS["overhead"]], (640, 480)), cv2.COLOR_RGB2BGR))
        return obs

    def run(self, task: str, seed: int, seconds: float, video: Path | None, log=print, realtime: bool = False) -> dict | None:
        eng = self.eng
        rng = np.random.default_rng(seed)
        eng.pause()
        eng.reset(randomize=True, seed=seed)
        try:
            ep = CandyExpert(eng).setup(task, rng)  # settles the candy, places the plate / parks the hand
        except NoCandy:
            eng.resume()
            return None
        self.world = World(eng, task, ep)
        if video is not None:
            video.parent.mkdir(parents=True, exist_ok=True)
            size, fps = ((960, 360), 30) if self.full_video else ((640, 480), 10)  # per control step / per observation
            self.writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            self.caption = f"{ep['instruction']}  (seed {seed})"
        eng.resume()
        t0 = time.time()
        self.sent = []
        self.runner.start(ep["instruction"], seconds=seconds, lockstep=not realtime)
        ok_for = 0.0
        while self.runner.running:
            time.sleep(0.25)
            if self.world.success():
                ok_for += 0.25
                if ok_for >= 2.0:  # held the success state for 2 s: done
                    self.runner.stop()
            else:
                ok_for = 0.0
        status = self.runner.status()
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        res = {"seed": seed, "task": task, "candy": ep["candy"], "instruction": ep["instruction"],
               "success": bool(self.world.success()), "phase": status.get("phase"), "error": status.get("error"),
               "chunks": status.get("chunks"), "wall_s": round(time.time() - t0, 1), **smoothness(self.sent),
               "candy_end": [round(float(v), 3) for v in eng.data.xpos[eng.model.body(ep["candy"]).id]]}
        self.world = None
        log(json.dumps(res))
        return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-server", default="127.0.0.1:8082")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--policy-type", default=None, help="pi0_fast / pi05 (default: the checkpoint's config.json)")
    ap.add_argument("--tasks", default="pick_place,give_human")
    ap.add_argument("--episodes", type=int, default=10, help="per task")
    ap.add_argument("--seconds", type=float, default=40.0, help="policy time per episode (sim seconds, lockstep)")
    ap.add_argument("--seed0", type=int, default=10_000)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--realtime", action="store_true",
                    help="physics keeps running while the policy infers (like the real arms); default: lockstep")
    ap.add_argument("--no-pad", action="store_true",
                    help="send 640x480 frames (the async server then stretches them to 224x224: the old, wrong behaviour)")
    ap.add_argument("--full-video", action="store_true", help="30 fps front + third-person video of every control step")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds to run (overrides --episodes/--seed0)")
    ap.add_argument("--out", default=None, help="results JSON (default: <video-dir>/results.json)")
    a = ap.parse_args(argv)

    eng = SimEngine(scene="candy")
    eng.start()
    port = int(a.policy_server.rsplit(":", 1)[1])
    ckpt = os.path.expanduser(a.checkpoint)
    cfg_path = Path(ckpt) / "config.json"
    policy_type = a.policy_type or (json.loads(cfg_path.read_text())["type"] if cfg_path.exists() else "pi0_fast")
    runner_kw = dict(server_address=a.policy_server, checkpoint=ckpt, policy_type=policy_type,
                     load_timeout=300.0, blocked_by=lambda: other_policy_clients(port))  # yield to the 1080
    video_dir = Path(os.path.expanduser(a.video_dir)) if a.video_dir else None
    episodes = Episodes(eng, runner_kw, full_video=a.full_video, pad=not a.no_pad)
    log = lambda msg: print(msg, flush=True)
    results = []
    try:
        for i, task in enumerate(a.tasks.split(",")):
            if a.seeds:  # explicit seeds, e.g. to film particular episodes
                for seed in map(int, a.seeds.split(",")):
                    vid = video_dir / f"{task}_{seed}.mp4" if video_dir else None
                    res = episodes.run(task, seed, a.seconds, vid, log, realtime=a.realtime)
                    if res is not None:
                        results.append(res)
                continue
            seed, done = a.seed0 + 1000 * i, 0  # each task its own seeds (else the same candy every time)
            while done < a.episodes:
                seed += 1
                vid = video_dir / f"{task}_{seed}.mp4" if video_dir else None
                res = episodes.run(task, seed, a.seconds, vid, log, realtime=a.realtime)
                if res is not None:
                    results.append(res)
                    done += 1
    finally:
        eng.stop()
    summary = {t: f"{sum(r['success'] for r in results if r['task'] == t)}/{sum(r['task'] == t for r in results)}"
               for t in a.tasks.split(",")}
    print(json.dumps({"checkpoint": a.checkpoint, "success": summary}, indent=2))
    out = a.out or (video_dir / "results.json" if video_dir else None)
    if out:
        Path(out).write_text(json.dumps({"checkpoint": a.checkpoint, "success": summary, "episodes": results}, indent=2))


if __name__ == "__main__":
    main()
