#!/usr/bin/env python
"""LoRA fine-tune of the pi0-FAST SO101 checkpoint on a LeRobot dataset, on the 5090 next to the live policy
server (:8081, ~16 GB).

- Run it with the separate training venv ~/pi-train (lerobot 0.6.1 + peft), NEVER ~/pi-serve (its
  site-packages carry the serving shims and must not change).
- Offline: the cached checkpoint config carries the 1080's shims (baked FAST tokenizer path,
  validate_action_token_prefix=false). A hub re-download would replace it.
- Caps this process's GPU memory so the policy server keeps what it needs.
- Maps the rig camera names (overhead/left_wrist/right_wrist) onto the checkpoint's (base_0_rgb/...).

    ~/pi-train/.venv/bin/python tools/finetune_pi0fast.py --repo-id local/sim_candy_v0 \\
        --dataset-root ~/lerobot/outputs/datasets/sim_candy_v0 --steps 3000 \\
        --out ~/lerobot/outputs/train/pi0fast_sim_candy_v0
"""
import argparse
import json
import os
import sys

CHECKPOINT = "delvingdeep/pi0fast-so101-bimanual"
RENAME = {"observation.images.overhead": "observation.images.base_0_rgb",
          "observation.images.left_wrist": "observation.images.left_wrist_0_rgb",
          "observation.images.right_wrist": "observation.images.right_wrist_0_rgb"}
# the Gemma language model's attention + MLP projections (pi0_fast defines no default LoRA targets)
LORA_TARGETS = r".*paligemma\.model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def local_base(dest: str) -> str:
    """A local view of the cached checkpoint for offline training: weights symlinked, small JSON configs
    copied (never edited in the shared HF cache, which the policy server reads), and the preprocessor's
    FAST action tokenizer pointed at the baked copy that config.json already uses."""
    from huggingface_hub import snapshot_download

    src = snapshot_download(CHECKPOINT, local_files_only=True)
    dest = os.path.expanduser(dest)
    os.makedirs(dest, exist_ok=True)
    baked = json.load(open(os.path.join(src, "config.json")))["action_tokenizer_name"]
    for name in os.listdir(src):
        s, d = os.path.join(src, name), os.path.join(dest, name)
        if os.path.lexists(d):
            os.remove(d)
        if name.endswith(".json"):
            cfg = json.load(open(s))
            steps = cfg.get("steps") if isinstance(cfg, dict) else None  # processor pipelines only
            for step in steps if isinstance(steps, list) else []:
                if "action_tokenizer_name" in step.get("config", {}):
                    step["config"]["action_tokenizer_name"] = baked
            json.dump(cfg, open(d, "w"), indent=2)
        else:
            os.symlink(os.path.realpath(s), d)
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id")
    ap.add_argument("--dataset-root")
    ap.add_argument("--out")
    ap.add_argument("--resume", default=None, help="a run's checkpoint dir (e.g. <out>/checkpoints/last) to continue")
    ap.add_argument("--episodes", default=None, help='train on these episodes only, e.g. "[2]" (overfit test)')
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="peak LR (LoRA wants more than the full-FT 2.5e-5)")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32, help="LoRA scale = alpha/rank; peft's default 8 at r16 halves updates")
    ap.add_argument("--save-freq", type=int, default=1000)
    ap.add_argument("--mem-fraction", type=float, default=0.42, help="of the GPU; :8081 holds ~16 GB of 32")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--base-dir", default="~/pi-train/base_ckpt", help="where the local checkpoint view is built")
    a = ap.parse_args(argv)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    import torch

    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    from lerobot.scripts.lerobot_train import main as train_main

    if a.resume:  # everything else comes from the run's saved train_config.json
        sys.argv = ["lerobot-train", "--resume=true",
                    f"--config_path={os.path.expanduser(a.resume)}/pretrained_model/train_config.json"]
        return train_main()
    if not (a.repo_id and a.dataset_root and a.out):
        ap.error("--repo-id, --dataset-root and --out are required (unless --resume)")
    base = local_base(a.base_dir)

    sys.argv = ["lerobot-train",
                f"--dataset.repo_id={a.repo_id}", f"--dataset.root={os.path.expanduser(a.dataset_root)}",
                f"--policy.path={base}", f"--rename_map={json.dumps(RENAME)}",
                "--policy.dtype=bfloat16", "--policy.gradient_checkpointing=true", "--policy.push_to_hub=false",
                f"--policy.optimizer_lr={a.lr}", f"--policy.scheduler_warmup_steps={a.warmup}",
                f"--policy.scheduler_decay_steps={a.steps}", f"--policy.scheduler_decay_lr={a.lr / 10}",
                "--peft.method_type=LORA", f"--peft.r={a.rank}", f"--peft.lora_alpha={a.lora_alpha}", f"--peft.target_modules={LORA_TARGETS}",
                "--peft.full_training_modules=[]",
                f"--batch_size={a.batch_size}", f"--steps={a.steps}", f"--save_freq={a.save_freq}",
                f"--num_workers={a.num_workers}", "--log_freq=25", "--wandb.enable=false",
                f"--output_dir={os.path.expanduser(a.out)}"]
    if a.episodes:
        sys.argv.append(f"--dataset.episodes={a.episodes}")
    train_main()


if __name__ == "__main__":
    main()
