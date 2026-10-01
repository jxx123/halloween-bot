# Real-time chunking (RTC) for the real arms

**The problem.** On the real robot the arms keep moving while the policy infers the next chunk
(~150–175 ms for π₀.₅ on the 5090, plus the network). Without RTC:
- the new chunk is planned from an observation that is already stale when it lands;
- the robot either stalls when the queue runs dry, or jumps where the old and new chunks disagree.

The sim eval used to hide this by pausing physics during inference (`lockstep`).

**What we run.** `tools/rtc_policy_server.py`: lerobot's async `PolicyServer` plus RTC for
flow-matching policies (π₀.₅, π₀). The new chunk is conditioned on the old chunk's unexecuted tail:
- the first `inference_delay` steps are frozen (they run during inference);
- the next `execution_horizon` steps are softly guided.

This uses lerobot's own `predict_action_chunk(inference_delay=…, prev_chunk_left_over=…)`. lerobot 0.6.1 only
drives RTC for a robot in the same process (`rollout/inference/rtc.py`); its async server ignores it.
This server keeps the async protocol, so clients work unchanged:

| piece | convention |
|---|---|
| observation timestep | the robot's last executed step (PolicyRunner and robot_client both send it) |
| new chunk | labelled from the next step; the client keeps only steps it hasn't run ("latest chunk wins") |
| RTC prefix | the previous chunk's normalized actions from that step on, padded/truncated to `execution_horizon` |
| delay (steps) | `ceil(max recent compute / dt) + --net-margin-steps` |

```bash
~/pi-train/.venv/bin/python tools/rtc_policy_server.py --port 8083            # RTC on
~/pi-train/.venv/bin/python tools/rtc_policy_server.py --port 8083 --no-rtc   # same timing, plain chunks
.venv/bin/python -m halloween_bot.sim.eval_policy --policy-server 127.0.0.1:8083 --realtime \
    --checkpoint ~/lerobot/outputs/train/pi05_sim_candy_v1/merged --tasks pick_place --seeds 1002,1005,1008
```

**Sim A/B** (π₀.₅ v1, real-time mode, 3 seeds, 20 s each; success 0 in both, since the model can't grasp yet):

| | p99 step jump | max jump | idle |
|---|---|---|---|
| no RTC | 13–30 units | 17–47 | 15–16% |
| RTC | 4.4–4.9 units | 8–17 | 14–16% |

Idle is the same in both runs. It's the sim client's loop overhead (capturing and letterboxing three frames
inside the 30 Hz loop), not queue starvation.

**On the real robot:**
- Use a PolicyRunner in non-lockstep mode, since real physics can't pause.
- Send letterboxed frames (`policy_runner.letterbox`; see finetune_v0.md, serving bug).
- Point the policy address at the RTC server.
- Set `--net-margin-steps` to the measured 1080→5090 round trip.
