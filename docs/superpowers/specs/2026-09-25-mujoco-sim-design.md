# MuJoCo sim + π₀-FAST policy tool for the bimanual SO101 Halloween bot

Date: 2026-09-25 · Branch: `sim` · Owner: 5090 Claude session (halloween-bot-73), with the
1080 workstation session ("Pumpkin face fullscreen") owning the real rig and `server.py`.

## Intent

Jinyu wants to try the Halloween bot against a simulated twin on the 5090 box
(`jinyuxie-OMEN…`, RTX 5090). The twin should behave like the real arms that the
jinyu-1080 box drives. The twin will be used to:

1. try the π₀-FAST SO101 bimanual checkpoint (`delvingdeep/pi0fast-so101-bimanual`) on
   the 5090,
2. let the upper-level Claude agent act by **either** moving joints directly **or**
   calling the π policy as a tool,
3. see all of it live in the existing web app.

Decisions Jinyu made (2026-09-25):
- **Sys-ID:** start from the Menagerie SO101 model plus the real calibration. Fit it to the
  existing 1080 teleop recording, then refine with a short real step-response run
  that the 1080 session executes after Jinyu approves.
- **Pi backend:** the existing lerobot gRPC `policy_server` in `~/pi-serve`. The sim acts as the robot client. The sim gets its own instance on :8081, because :8080 serves the 1080's real-robot client and a policy server handles one client at a time.
- **Real robot:** the policy runner is a shared module. The 1080 session wires it into
  `server.py` and tests it on hardware. This session does not edit `server.py`.
- **Timing:** real-time by default (the sim keeps moving during inference, like the real
  robot). A lockstep flag pauses physics during inference.

## Success criteria

1. `python -m halloween_bot.ctl start --sim` brings up a sim server on :8399 with the
   **same HTTP API** as `server.py`. The **unchanged** web app shows the rendered
   `overhead`, `left_wrist` and `right_wrist` streams. `ctl.py state|move|cam` work as-is.
2. The normalized joint units (`-100..100`, gripper `0..100`) mean the same pose in sim and on
   the real arm. They are derived from the 1080's calibration files and checked visually
   against matched real reference frames.
3. Dynamics: the fitted model has lower tracking error than the Menagerie defaults when
   replaying the real teleop recording. Target: body-joint RMSE ≤ 3 normalized units. The
   before/after numbers and overlay plots are reported either way.
4. `ctl.py policy "Grasp the toy and place it in the basket." --seconds 30` runs π₀-FAST
   against the sim end to end, with `--lockstep` as an option. **Task success in sim is
   not a criterion.** The checkpoint was trained on real images from a different rig, so
   the visual gap is expected.
5. The webapp's `claude` and `pumpkin` agents can move joints and call the policy.
   `CLAUDE.md` tells them when to use which.
6. `policy_runner.py` has no sim-specific code, so `server.py` can adopt it unchanged.

Non-goals: making π succeed in sim, domain randomization or training, photoreal
rendering, modifying `server.py`.

## Architecture

```
Browser ─► webapp :8500 (unchanged API; sim-aware dashboard)
               │ /api/state /api/policy /api/sim/reset      MJPEG from :8399/stream
               ▼
   sim server :8399  (halloween_bot/sim/server.py — same API as server.py, + /sim/*, /policy)
      ├─ physics thread: MuJoCo, real-time, position actuators, per-tick clamp like lerobot
      ├─ render thread:  EGL offscreen, overhead / left_wrist / right_wrist / scene
      └─ PolicyRunner ── gRPC ──► lerobot policy_server :8080 (~/pi-serve venv, GPU)
                                   π₀-FAST, 3 cams @224² + 12-d state → 50×12 action chunk
Claude agent (webapp → `claude -p`, cwd=repo) ─► python -m halloween_bot.ctl move|cam|policy
```

### 1. Model: `halloween_bot/sim/model.py` + `assets/`

- Vendors the Menagerie `robotstudio_so101` MJCF and meshes (Apache-2.0, LICENSE copied).
- Builds the scene with `mujoco.MjSpec`. Two SO101 copies are attached with prefixes `left_`
  (orange) and `right_` (blue), so joint and actuator names match lerobot keys minus
  `.pos`.
- World frame: +x forward (toward the toys and camera), +y robot-left, +z up. Both bases
  sit on the back edge at x = 0, with left at y = +0.175 and right at y = −0.175
  (≈35 cm apart). Both face +x.
- Scene: a black table, two dark-gray mats (55×40 cm each), a white basket (≈20 cm
  diameter × 12 cm, built from a ring of box walls) at x≈0.20, y≈0, and two toys (soft
  compound bodies, ≈8 cm) at x≈0.25–0.35 on free joints. There's also an optional mouse
  obstacle at the right edge of the orange arm's reach.
- Cameras:
  - `overhead`: front-elevated view, not top-down. About x=0.60, z=0.42, pitched down
    ≈35°, looking back at the arms, 1280×720, horizontal FOV ≈70°.
  - `left_wrist` / `right_wrist`: Menagerie's `wrist_cam` on each gripper, 640×480.
  - `scene`: third-person view for the dashboard.
  All camera poses are tuned by eye against the 1080 reference frames.
- Actuator and joint parameters come from `halloween_bot/sim/params.json` (sys-ID output).
  If that file is missing, Menagerie `sts3215` defaults apply.

### 2. Calibration mapping: `halloween_bot/sim/calib.py`

- Loads the 1080's `so_follower/bimanual_{left,right}.json`, vendored as
  `halloween_bot/sim/calibration/`.
- Body joints: `raw = min + (n+100)/200·(max−min)`, then `rad = s·(raw − 2047)·2π/4096`.
- Gripper: `raw = min + n/100·(max−min)`, same raw→rad formula.
- `s ∈ {+1,−1}` per joint, plus an optional per-joint zero offset. These are the only
  free parameters. They're settled once by rendering the sim at the reference `/state` and
  comparing to the matched real frames.
- The inverse (rad → normalized) is used for `/state`.
- Unit-tested: round-trip, and every calibrated range maps inside the MJCF joint limits.

### 3. Sim server: `halloween_bot/sim/server.py`

Same endpoints and JSON shapes as `server.py`:

- `GET /state` returns `{"ok", "positions", "teleop": false, "sim": true, "policy": {...}}`.
- `POST /move` copies the real server exactly: 30 Hz linear interpolation, and each tick's
  goal is clamped to ±20 normalized units from the present position (lerobot
  `max_relative_target`). Unknown keys and duration clamping behave identically.
- `GET /cam?name=` writes a JPEG to the same `FRAMES_DIR`.
- `GET /stream?name=` serves MJPEG at 15 fps and also accepts `scene`.
- `POST /teleop` returns `{"ok": false, "error": "no leader arms in sim"}`.
- `POST /cam_restart` is a no-op that returns ok.
- `POST /stop` exits.

New endpoints:

- `POST /sim/reset {"randomize": bool}` puts the arms back at the rest pose (pick_toys
  `PARK`: pan −5, lift −86, elbow 95, wrist_flex 45, wrist_roll −10, gripper 40, both
  arms) and re-places the toys.
- `POST /trajectory {"points": [{targets}], "hz": 30}` plays a commanded sequence and
  returns a per-tick trace `[{t, cmd, present}]`. The real server gets the same
  endpoint (1080 side) for sys-ID A/B.
- `/policy` endpoints come from PolicyRunner (§4).

Concurrency and binding:

- Physics runs in its own thread at the model timestep (5 ms), paced to wall clock.
  Commands only set `ctrl`.
- `/move` returns 409 while a policy runs, the same way it does during teleop.
- Rendering runs in one thread that owns the EGL context. It publishes the latest frame
  per camera, and readers never touch MuJoCo.
- Binds 127.0.0.1 plus the tailnet address, same as the real server. `--port` overrides.

### 4. Policy runner: `halloween_bot/policy_runner.py` (shared, robot-agnostic)

```python
PolicyRunner(get_observation: () -> dict,   # {"left_shoulder_pan.pos": f, ..., "base_0_rgb": HxWx3 uint8, ...}
             send_action: (dict) -> None,   # 12 joint targets, normalized
             server_address="127.0.0.1:8080",
             checkpoint="delvingdeep/pi0fast-so101-bimanual", policy_type="pi0_fast",
             fps=30, actions_per_chunk=50, chunk_size_threshold=0.5,
             pause: (() -> None) | None = None, resume: (() -> None) | None = None)
.start(task, seconds, lockstep=False) · .stop() · .status() -> dict
```

- It speaks the lerobot 0.6.1 async protocol, reusing lerobot's own `RemotePolicyConfig`,
  `TimedObservation`/`TimedAction`, `services_pb2(_grpc)` and `send_bytes_in_chunks`:
  `Ready` → `SendPolicyInstructions` → loop of `SendObservations` / `GetActions`.
  The queue semantics (overlapping timesteps, latest chunk wins) match `robot_client`.
- Camera renames live in the caller's `get_observation`:
  `overhead→base_0_rgb`, `left_wrist→left_wrist_0_rgb`, `right_wrist→right_wrist_0_rgb`.
- Lockstep, which needs `pause`/`resume`: observations are only sent with the queue
  empty (`must_go`), and physics pauses until the chunk arrives.
- It never starts the policy server. If the server is unreachable, `start()` fails fast
  with a clear error.
- Endpoints, mounted by any server:
  - `POST /policy {"task", "seconds", "lockstep"}`
  - `POST /policy/stop`
  - `GET /policy` → `{running, task, elapsed, chunks, last_latency_s, error}`

### 5. CLI: `halloween_bot/ctl.py` (additive)

- `start --sim` launches `halloween_bot.sim.server` instead of `server.py`.
- `policy "<task>" [--seconds 30] [--lockstep] [--no-wait]` blocks and prints progress
  until the run ends, then prints the final status.
- `policy-stop`.
- `sim-reset [--randomize]`.
- `cam` also accepts `overhead` and `scene`.

### 6. Web app (additive)

- `HALLOWEEN_ROBOT_API` env var (default `http://127.0.0.1:8399`).
- Proxies for `/api/policy` (GET/POST), `/api/policy/stop` and `/api/sim/reset`.
- Dashboard, when `state.sim` is true:
  - a **SIM** badge;
  - a `scene` stream panel;
  - a policy bar: task input (defaults to the checkpoint's training prompt), seconds,
    lockstep toggle, ▶ run π / ■ stop, and a live status line;
  - a ⟲ reset button;
  - the teleop button is hidden.

### 7. Agent wiring

- `CLAUDE.md` becomes machine-agnostic: one command form per machine (5090:
  `.venv/bin/python -m halloween_bot.ctl`; 1080: the lerobot conda env).
- New section **"Two ways to act"**:
  - Direct joint moves for gestures, waving, pointing, and anything precise or scripted.
  - `ctl.py policy "Grasp the toy and place it in the basket."` for the pick-and-place
    skill π was trained on.
  - Always look (overhead cam) before and after.
  - Stop the policy if the arms go somewhere unsafe.
- New section **"Sim mode"**: `/state` has `"sim": true`, droop rules don't apply, and
  `sim-reset` is available.
- `agents.py` is unchanged. The agent already has Bash and Read, and its cwd is the repo root.

### 8. System identification: `halloween_bot/sim/sysid.py`

- Data:
  - **(a)** The 1080 teleop episode, already pushed to `~/so101_bimanual_test_dataset/`
    (LeRobot v3): 778 frames @30 Hz of `action` (commanded) and `observation.state`
    (present).
  - **(b)** Real traces from `/trajectory` sequences, generated by
    `sysid_collect.py`: small steps and chirps around mid-poses on one joint at a time,
    never full-range steps at the lift extremes. The 1080 session runs it through its
    server after Jinyu's go, once it has confirmed the workspace is clear.
- Replay: headless, faster than real time. Each 30 Hz command is clamped like lerobot,
  then physics steps for 1/30 s. The sim `present` is compared to the recorded `present`
  at t+1.
- Fit: per-joint `kp`, `kv`, `frictionloss`, `armature`, `forcerange`, shared across
  both arms at first. Uses `scipy.optimize.least_squares` on normalized-unit residuals.
  Holdout: the last 20% of each trace.
- Output: `params.json` plus `sysid_report.md` with RMSE per joint (Menagerie vs fitted)
  and overlay PNGs.
- Known real targets: elbow sags about +5 units at horizontal extension, and shoulder_lift
  about 1–2 units (P=16, D=32 firmware gains).

### 9. Environment (5090)

- The repo gets its own `.venv` (uv, Python 3.12): `uv pip install -e ".[sim]"`, where
  the new extra is `sim = ["mujoco>=3.3", "scipy", "pandas", "pyarrow"]`.
- lerobot is pinned `==0.6.1` for sim use, matching `~/pi-serve`. That match is required
  because observations and actions are pickled across the gRPC link.
- Rendering uses `MUJOCO_GL=egl`.
- `~/pi-serve` carries hand-applied shims: a baked FAST tokenizer, `pi0_fast` whitelisted in
  the async constants, and `validate_action_token_prefix=false` in the cached checkpoint
  config. Its site-packages are **never edited or reinstalled** from this project, because
  uv hardlinks them into the wheel cache. The client venv never touches that file.
- `run_sim.sh` starts the policy server (`~/pi-serve`), the sim server and the webapp,
  each with a log under `~/lerobot/outputs/claude_robot/`.

## Collaboration with the 1080 session

- The 1080 session owns `server.py` and all real-robot motion. This session never commands
  the real arms.
- 1080 deliverables, requested:
  1. Matched reference frames plus `/state` at rest. **Done:** in `~/sim_ref/` on the 5090.
     A v2 without the bunny keychain on the right wrist cam is pending.
  2. Adding a `/trajectory` endpoint and the `PolicyRunner` mount to `server.py`.
  3. Running `sysid_collect.py` after Jinyu's explicit go.
- Git: work happens on branch `sim`. The shared files (`ctl.py`, `webapp.py`,
  `dashboard.html`, `CLAUDE.md`) get additive edits only, and any change to them is
  announced to the peer.

## Testing

- **pytest (fast, no GPU policy):**
  - calibration round-trip and limits;
  - the model builds, with 12 actuators named to match lerobot keys;
  - sim `/move` reaches targets within 2 units and obeys the per-tick clamp;
  - `/state` key set equals the real `action_features`;
  - PolicyRunner against an in-process fake gRPC servicer: queue aggregation, stop,
    timeout, unreachable-server error, and lockstep pause/resume calls.
- **Integration (5090):**
  - policy server + sim server + `ctl.py policy … --seconds 20`: chunks received and
    joints moving;
  - a scene-cam MP4 saved for Jinyu;
  - the webapp checked in the browser: streams, SIM badge, policy buttons;
  - one `claude -p` instruction that uses a direct move and one that calls `policy`.
- **Sys-ID:** the report numbers are the evidence for criterion 3.

## Risks

- The Menagerie zero pose may not equal the lerobot calibration mid pose, and joint signs
  may differ. Handled by the per-joint `s` and offset, checked against reference frames.
- π in sim will likely wander (visual gap and a different training rig). This is stated
  up front and is not a failure.
- The 1.6 s inference against 1.67 s chunks leaves the queue barely fed in real-time mode.
  This mirrors the real robot. Lockstep mode exists for clean runs.
- The policy server (~7 GB) and the other GPU tenants: 23 GB is currently free.
