#!/usr/bin/env python
"""Merge a LoRA fine-tune (lerobot-train --peft) into a full pi0-FAST checkpoint that the ~/pi-serve policy
server can load (it has no peft). Keeps the fine-tune's pre/post-processors: their normalizer stats are
the training dataset's, not the base checkpoint's.

    ~/pi-train/.venv/bin/python tools/merge_lora.py \\
        ~/lerobot/outputs/train/pi0fast_sim_candy_v0/checkpoints/last/pretrained_model \\
        ~/lerobot/outputs/train/pi0fast_sim_candy_v0/merged
"""
import argparse
import json
import os
import shutil


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adapter", help="the fine-tune's pretrained_model dir (adapter_config.json, config.json, processors)")
    ap.add_argument("out")
    ap.add_argument("--device", default="cpu", help="cpu keeps the GPU free for the live policy server")
    a = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from peft import PeftModel

    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

    adapter, out = os.path.expanduser(a.adapter), os.path.expanduser(a.out)
    base = json.load(open(os.path.join(adapter, "adapter_config.json")))["base_model_name_or_path"]
    policy = PI0FastPolicy.from_pretrained(base)
    policy.to(a.device)
    merged = PeftModel.from_pretrained(policy, adapter).merge_and_unload()
    merged.config.use_peft = False
    os.makedirs(out, exist_ok=True)
    merged.save_pretrained(out)
    # the fine-tune's processors (dataset normalizer stats, rename map, baked tokenizer path)
    for name in os.listdir(adapter):
        if name.startswith(("policy_preprocessor", "policy_postprocessor")):
            shutil.copy2(os.path.join(adapter, name), os.path.join(out, name))
    cfg = json.load(open(os.path.join(out, "config.json")))
    print(f"merged -> {out}  (type={cfg['type']}, use_peft={cfg.get('use_peft')}, "
          f"inputs={list(cfg['input_features'])})")


if __name__ == "__main__":
    main()
