#!/usr/bin/env python
"""Quick sanity check of a pi0-FAST / pi05 checkpoint on its training data, without the sim or a policy server:
predict the action chunk at a few frames and print it next to the demo's (right arm; --arm left).
A model that has learned the demos tracks them; one that hasn't predicts "stay put" or nonsense.

Run with the ~/pi-train venv (it has the pi0_fast tokenizer shims):
    ~/pi-train/.venv/bin/python tools/check_policy_offline.py ~/lerobot/outputs/train/<run>/merged \\
        --repo-id local/sim_candy_v0 --root ~/lerobot/outputs/datasets/sim_candy_v0
"""
import argparse
import os


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--repo-id", default="local/sim_candy_v0")
    ap.add_argument("--root", default="~/lerobot/outputs/datasets/sim_candy_v0")
    ap.add_argument("--episodes", default="0,1,2")
    ap.add_argument("--offsets", default="0,60,150")
    ap.add_argument("--arm", choices=("left", "right"), default="right")
    ap.add_argument("--mem-fraction", type=float, default=0.42)
    ap.add_argument("--device", default="cuda", help="cpu works (slow) when training holds the GPU")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import numpy as np
    import torch

    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    import json

    from lerobot.policies.factory import get_policy_class

    ckpt = os.path.expanduser(a.checkpoint)
    kind = json.load(open(os.path.join(ckpt, "config.json")))["type"]
    policy = get_policy_class(kind).from_pretrained(ckpt).to(a.device).eval()
    policy.config.device = a.device
    pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt,
                                         preprocessor_overrides={"device_processor": {"device": a.device}})
    ds = LeRobotDataset(a.repo_id, root=os.path.expanduser(a.root))
    cols = slice(0, 6) if a.arm == "left" else slice(6, 12)
    errs = []
    for e in map(int, a.episodes.split(",")):
        start, end = int(ds.meta.episodes[e]["dataset_from_index"]), int(ds.meta.episodes[e]["dataset_to_index"])
        for off in map(int, a.offsets.split(",")):
            if start + off + 50 > end:
                continue
            f = ds[start + off]
            batch = {k: (v.unsqueeze(0) if hasattr(v, "unsqueeze") else v) for k, v in f.items()}
            batch["task"] = [f["task"]]
            with torch.no_grad():
                out = post(policy.predict_action_chunk(pre(batch)))
            pred = out[0].float().cpu().numpy()
            demo = np.stack([ds[start + off + k]["action"].numpy() for k in range(50)])
            err = float(np.abs(pred[:, cols] - demo[:, cols]).mean())
            errs.append(err)
            print(f"ep{e} t{off} [{f['task'][:45]}]  mean |pred-demo| {a.arm} arm: {err:.1f} units")
            for k in (0, 20, 40):
                print(f"    t+{k:2d} pred {np.round(pred[k, cols], 0).tolist()}  demo {np.round(demo[k, cols], 0).tolist()}")
    print(f"overall mean |pred-demo|: {np.mean(errs):.1f} units")


if __name__ == "__main__":
    main()
