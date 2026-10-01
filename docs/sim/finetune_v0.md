# π₀-FAST fine-tune on sim candy data: v0 (2026-09-27/28)

A pipeline check: fine-tune the SO101 π₀-FAST checkpoint on `sim_candy_v0` (100 scripted episodes:
pick_place and give_human) and evaluate it closed-loop in the twin.

## Setup

- **Training venv:** `~/pi-train`: lerobot 0.6.1, peft, and the 1080's two tokenizer shims, copied by
  tmpfile + rename. `~/pi-serve` is never touched.
- **Training:** `tools/finetune_pi.py`:
  - LoRA r=16, alpha=32, on the Gemma language model's attention and MLP projections (20M trainable
    params). LR 2e-4, warmup 500, cosine decay, batch 4, bf16, gradient checkpointing.
  - Offline, GPU memory capped at 42%, so it trains next to the live :8081 server.
- **Serving:** `tools/merge_lora.py` merges into a full checkpoint for `~/pi-serve`, which has no peft.
  It's served from a second policy server on :8082.
- **Checks:**
  - Offline: `tools/check_policy_offline.py`, predictions on training frames.
  - Closed loop: `halloween_bot/sim/eval_policy.py`, seeded candy episodes (seeds 10000+, never
    recorded), lockstep, the recorder's success tests.

## Results

| run | steps | loss | offline mean \|pred−demo\| | sim pick_place | sim give_human |
|---|---|---|---|---|---|
| base checkpoint | – | – | – | 0/2 | 0/2 |
| alpha 8, LR 1e-4 | 4k | 7.7 → 2.6 | – | 0/5 (never moves) | – |
| alpha 8, LR 1e-4 | 5.7k (stopped) | 2.33 | 29.0 | – | – |
| **alpha 32, LR 2e-4** | **20k** (~1.7 epochs) | **7.7 → 0.95** | **10.9** (mid-reach 1–2) | **0/10** | **0/10** |
| **π₀.₅ (lerobot/pi05_base)**, alpha 32, LR 2e-4 | 20k | 0.16 → 0.034 (flow MSE) | 9.3 (start of reach 9.7 vs 34.6) | 0/10 | 0/10 |
| **π₀.₅ on sim_candy_v1** (1000 eps) | 40k | 0.32 → 0.054 | 8.7–9.2 | 0/10 (0/4 on training seeds) | 0/10 |

With the 20k model the arm moves in 18/20 sim episodes. It reaches into the bowl toward the named
candy, misses the grasp, and returns to rest. It has learned the shape of the task but not
millimetre grasp precision.

With π₀.₅ the arms are much more active. It often reaches with one arm, retracts, then tries the other: the
which-arm ambiguity shows up as sequential attempts instead of FAST's averaged no-motion. It still misses
the grasp. π₀.₅ needs no FAST tokenizer or shims: it trains from `lerobot/pi05_base` and is served by stock
`~/pi-serve` (`tools/finetune_pi.py --policy pi05`).

## Findings

- **The pipeline is correct.** An overfit test on one episode reproduces static chunks to <1 unit.
  FAST chunks are ~33 tokens (max 58 of 256), and decode errors: none.
- **LoRA alpha.** peft's default `lora_alpha=8` at r=16 scales updates by 0.5; motion was barely
  learned. alpha 32 fixed that.
- **Failure mode.** An undertrained model predicts "stay at rest" from the start pose: the
  motion-rich start of a reach is learned last. It is not an idle-data problem: 94% of the acting
  arm's 50-step chunks move.
- **Decode spikes.** A malformed token can decode to a huge value (seen once: lift 68,529). The
  per-tick ±20 clamp and the joint limits contain it, in sim and on the real server.

## Serving bug (found 2026-09-29, fixed in commit 0228fe0)

lerobot's async policy server (`async_inference/helpers.resize_robot_observation_image`) resizes every incoming
image straight to 224×224, stretching our 4:3 frames. In training the policy letterboxes them itself
(`resize_with_pad`: aspect kept, 28 black rows top and bottom), and it skips that step when serving because the
shape already matches. So every served policy above saw squashed images it never trained on.
`policy_runner.letterbox()` reproduces resize_with_pad exactly, and the sim server and eval now send letterboxed
224×224 frames. The rows above were measured before the fix. With the fix, π₀.₅ v1 is still 0/4 on training
seeds: on those exact frames it predicts the right reach offline, but served it often samples "both arms
wait". The right arm rests in about half the demos (left-arm episodes), so the arm choice is the weak mode.

## Next

- **v2 data:** consistent arm choice (`record.py --arm-rule midline`); v1 had coin flips in 14.9% of episodes
  and a moving bowl reference.

- **More sim data:** ~1000 episodes (~2 h to record; the screening makes yield irrelevant), then a
  longer run. Precise grasping from 100 demos is optimistic.
- **Real demos:** real teleop demos from the 1080's `candy_record.sh`, mixed in for sim2real. Stamp the
  sim data `robot_type=bi_so_follower` so `aggregate_datasets()` merges them.
- **Real-arm eval:** only once sim success is non-trivial, and only on Jinyu's go in the 1080 session.
