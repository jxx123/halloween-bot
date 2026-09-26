# Claude ⇄ Bimanual SO101 control bridge

You (a Claude Code agent) control two SO101 follower arms through `ctl.py`. The arms are either
the real ones on jinyu-1080 or their MuJoCo twin on the 5090. A persistent server on :8399
holds the robot or the sim, and the arms keep torque and hold their pose between your commands.

## Commands

Run from the repo root (this directory). Pick the python for the machine you're on:

```bash
PY=.venv/bin/python                        # 5090 (MuJoCo twin)
PY=~/miniconda3/envs/lerobot/bin/python    # jinyu-1080 (real arms)

$PY -m halloween_bot.ctl start             # real robot (arms hold pose, torque ON)
$PY -m halloween_bot.ctl start --sim       # MuJoCo twin instead (5090) — same API
$PY -m halloween_bot.ctl state             # all joint positions (+ "sim": true on the twin)
$PY -m halloween_bot.ctl cam overhead      # capture frame -> JPEG path (Read it to see)
$PY -m halloween_bot.ctl cam left_wrist    # also right_wrist; "scene" = third-person view (sim only)
$PY -m halloween_bot.ctl move '{"left_gripper.pos": 50}'       # smooth 2 s move
$PY -m halloween_bot.ctl move '{"right_shoulder_pan.pos": 10, "right_elbow_flex.pos": -15}' --duration 3
$PY -m halloween_bot.ctl policy "Grasp the toy and place it in the basket." --seconds 30
$PY -m halloween_bot.ctl policy-stop
$PY -m halloween_bot.ctl stop              # real: torque OFF + server exit (see safety rule 4)
```

The whole 5090 stack (π policy server, sim server, web app) comes up with `./run_sim.sh`.

## Two ways to act

1. **Direct joint moves** (`ctl move`). Use them for gestures, waving, pointing, presenting
   candy, dancing, and anything scripted or precise. You decide every joint target, and you
   can see the result.
2. **The π₀-FAST policy** (`ctl policy "<task>" --seconds N`). This is a learned
   vision-language-action model that drives both arms from the three cameras. Its skill is
   what it was trained on: **"Grasp the toy and place it in the basket."** Use it to put a
   toy in the basket.
   - `ctl policy` blocks and prints progress (`loading → running → done`). The first run after
     a policy-server restart spends ~50 s loading the model.
   - `/move` is refused while the policy runs. `ctl policy-stop` aborts it, and the arms hold
     wherever it left them.
   - On a server that hasn't wired π yet, `ctl policy` fails with a clear error. Fall back to
     direct moves.

Either way: **look before and after** (overhead frame, see below). If one π run made no
progress, don't loop it. Say so, or switch to direct moves. π is a statistical policy trained
on a different rig, so it can miss.

## Joint keys and ranges

`{left|right}_{shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll}.pos` — normalized
**-100..100** (0 ≈ middle of that joint's calibrated range).
`{left|right}_gripper.pos` — **0..100** (0 = closed, 100 = open).
Left arm = orange, right arm = blue. Rest pose (both arms): pan -5, lift -86, elbow 95,
wrist_flex 45, wrist_roll -10, gripper 40.
Cameras:
- `left_wrist`, `right_wrist`: wrist-mounted, 640×480.
- `overhead`: Logitech C922, 1280×720. Despite the name, it's a front-elevated view looking
  back at the arms. Use this one to locate objects; the wrist cams suffer parallax at close range.

## Always-fresh overhead view

`~/lerobot/outputs/claude_robot/latest/overhead.jpg` is refreshed every few seconds by the
web app's scene watcher whenever the robot server is up (real or sim). **Read it first for
any spatial task.** It's the current scene, and no ctl.py call is needed. Orientation: the camera
faces the arms, so image-left = blue/RIGHT arm and image-right = orange/LEFT arm.

## Sim mode (MuJoCo twin on the 5090)

`ctl state` shows `"sim": true`. Same keys, units, speed caps and cameras as the real rig. The
joint mapping comes from the real calibration, and the dynamics are fit to real
recordings.
- Nothing can break. The droop/stop rules below don't apply, but move the way you would on
  the real arms, because that's the point of the twin.
- `$PY -m halloween_bot.ctl sim-reset` puts the arms back at rest and re-places the toys.
  `--randomize` scatters the toys.
- `ctl cam scene` gives a third-person view that only exists in sim.

## Safety rules (real robot) — follow these

1. **Look before you move**: capture both wrist cams and read `state` before the first move
   and after anything unexpected.
2. **Small steps**: change body joints by ≤ 20-30 units per move until you have visually
   confirmed the workspace is clear. The server also caps per-tick deltas.
3. **Gripper**: close gently. Go to the value where the object is held, not 0. Holding a
   stalled gripper trips the motor's overload protection.
4. **Before `stop`**: torque releases and the arms DROOP. Move both arms to the rest pose
   near the table first.
5. One bus owner at a time: the server refuses to start while `lerobot-teleoperate` or
   `lerobot-record` runs (`pkill -INT -f 'lerobot-(teleoperate|record)'` to free the buses).

## Troubleshooting

- Transient `ConnectionError: ... no status packet` on start → retry once. It's a dropped packet.
- Totally silent bus → check the power barrel jack on that arm (the orange follower's has been
  loose before).
- Logs: `~/lerobot/outputs/claude_robot/server.log` (real), `sim_server.log`,
  `policy_server_sim.log` and `webapp.log` (5090).

## Pumpkin Bot persona (terminal)

`app/persona.md` is the kid-facing Halloween voice the web app's "pumpkin" agent uses.
A plain `claude` session here does NOT load it. To talk to Pumpkin Bot from the terminal:

```bash
./pumpkin.sh                 # interactive session with the persona appended
./pumpkin.sh -p "say hello"  # one-shot
```
