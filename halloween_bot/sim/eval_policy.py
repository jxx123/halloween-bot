#!/usr/bin/env python
"""Evaluate a pi0-FAST checkpoint on the candy tasks in the MuJoCo twin.

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

import cv2
import numpy as np

from ..policy_runner import PI0FAST_CAMERAS, PolicyRunner
from . import candy as C
from .calib import KEYS
from .candy_expert import CandyExpert, NoCandy
from .engine import SimEngine
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


class Episodes:
    """One PolicyRunner for the whole evaluation (it instructs the server once, so the model loads once);
    the observation hook drives whichever episode is current."""

    def __init__(self, eng: SimEngine, runner_kw: dict):
        self.eng, self.world, self.writer = eng, None, None
        self.runner = PolicyRunner(self._observe, eng.send_action, KEYS, policy_features(), pause=eng.pause,
                                   resume=eng.resume, **runner_kw)

    def _observe(self):
        if self.world is not None:
            self.world.tick()
        obs = policy_observation(self.eng)
        if self.writer is not None:
            self.writer.write(cv2.cvtColor(obs[PI0FAST_CAMERAS["overhead"]], cv2.COLOR_RGB2BGR))
        return obs

    def run(self, task: str, seed: int, seconds: float, video: Path | None, log=print) -> dict | None:
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
            self.writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))  # per observation
        eng.resume()
        t0 = time.time()
        self.runner.start(ep["instruction"], seconds=seconds, lockstep=True)
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
               "chunks": status.get("chunks"), "wall_s": round(time.time() - t0, 1),
               "candy_end": [round(float(v), 3) for v in eng.data.xpos[eng.model.body(ep["candy"]).id]]}
        self.world = None
        log(json.dumps(res))
        return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-server", default="127.0.0.1:8082")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tasks", default="pick_place,give_human")
    ap.add_argument("--episodes", type=int, default=10, help="per task")
    ap.add_argument("--seconds", type=float, default=40.0, help="policy time per episode (sim seconds, lockstep)")
    ap.add_argument("--seed0", type=int, default=10_000)
    ap.add_argument("--video-dir", default=None)
    ap.add_argument("--out", default=None, help="results JSON (default: <video-dir>/results.json)")
    a = ap.parse_args(argv)

    eng = SimEngine(scene="candy")
    eng.start()
    port = int(a.policy_server.rsplit(":", 1)[1])
    runner_kw = dict(server_address=a.policy_server, checkpoint=os.path.expanduser(a.checkpoint),
                     load_timeout=300.0, blocked_by=lambda: other_policy_clients(port))  # yield to the 1080
    video_dir = Path(os.path.expanduser(a.video_dir)) if a.video_dir else None
    episodes = Episodes(eng, runner_kw)
    log = lambda msg: print(msg, flush=True)
    results = []
    try:
        for i, task in enumerate(a.tasks.split(",")):
            seed, done = a.seed0 + 1000 * i, 0  # each task its own seeds (else the same candy every time)
            while done < a.episodes:
                seed += 1
                vid = video_dir / f"{task}_{seed}.mp4" if video_dir else None
                res = episodes.run(task, seed, a.seconds, vid, log)
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
