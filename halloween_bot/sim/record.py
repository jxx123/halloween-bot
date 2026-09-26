#!/usr/bin/env python
"""Record scripted pick-and-place demos in the MuJoCo twin as a LeRobot dataset.

Same layout as a real rig recording: 12-d observation.state / action in the rig's normalized
units (KEYS order) and the rig's three cameras (overhead cropped to the C922's 4:3 mode, both
wrists) at 30 fps. Each attempt: reset with randomized toys, let them settle, run the Expert; only
successful attempts are saved unless --keep-failures.

    python -m halloween_bot.sim.record --episodes 20 --root ~/lerobot/outputs/datasets/sim_pick_place
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..policy_runner import DEFAULT_TASK, fit_image
from .calib import KEYS
from .engine import SimEngine
from .expert import Expert

CAMERAS = ("overhead", "left_wrist", "right_wrist")
FPS = 30


def features(image_hw: tuple[int, int]) -> dict:
    h, w = image_hw
    vec = {"dtype": "float32", "shape": (len(KEYS),), "names": list(KEYS)}
    return {"observation.state": vec, "action": dict(vec),
            **{f"observation.images.{c}": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
               for c in CAMERAS}}


def record(episodes: int, root, repo_id: str = "local/sim_pick_place", seed0: int = 0, task: str = DEFAULT_TASK,
           image_hw: tuple[int, int] = (480, 640), keep_failures: bool = False, max_attempts: int | None = None,
           log=print) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.create(repo_id=repo_id, fps=FPS, features=features(image_hw), root=root,
                               robot_type="sim_bi_so101", use_videos=True, image_writer_threads=4)
    eng = SimEngine()
    saved = attempts = 0
    results = []
    max_attempts = max_attempts or 3 * episodes
    t0 = time.time()
    try:
        while saved < episodes and attempts < max_attempts:
            seed = seed0 + attempts
            attempts += 1
            eng.reset(randomize=True, seed=seed)
            expert = Expert(eng)

            def tick(action: dict):  # called before the action executes: (obs_t, action_t)
                obs = eng.read_positions()
                frame = {"observation.state": np.array([obs[k] for k in KEYS], np.float32),
                         "action": np.array([action[k] for k in KEYS], np.float32), "task": task}
                for cam in CAMERAS:
                    frame[f"observation.images.{cam}"] = fit_image(eng.get_frame(cam), image_hw)
                ds.add_frame(frame)

            res = expert.run(np.random.default_rng(seed), on_tick=tick)
            results.append({"seed": seed, **res})
            if res["success"] or keep_failures:
                ds.save_episode()
                saved += 1
            else:
                ds.clear_episode_buffer()
            log(f"attempt {attempts} seed {seed}: {res['toy']} with the {res['arm']} arm, "
                f"{'SUCCESS' if res['success'] else 'fail'} ({res['frames']} frames) -> saved {saved}/{episodes}")
    finally:
        ds.finalize()
    stats = {"saved": saved, "attempts": attempts, "success_rate": round(sum(r["success"] for r in results) / max(1, attempts), 3),
             "seconds": round(time.time() - t0, 1), "root": str(root), "results": results}
    (Path(root) / "meta" / "sim_record.json").write_text(json.dumps(stats, indent=2))
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--root", default=str(Path.home() / "lerobot/outputs/datasets/sim_pick_place"))
    ap.add_argument("--repo-id", default="local/sim_pick_place")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default=DEFAULT_TASK)
    ap.add_argument("--keep-failures", action="store_true")
    a = ap.parse_args(argv)
    if Path(a.root).exists():
        raise SystemExit(f"{a.root} exists; pick a new --root (LeRobot datasets are created fresh)")
    stats = record(a.episodes, a.root, a.repo_id, a.seed, a.task, keep_failures=a.keep_failures)
    print(json.dumps({k: v for k, v in stats.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
