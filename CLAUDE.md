# Claude ⇄ Bimanual SO101 control bridge

You (a Claude Code agent) can control two SO101 follower arms through `ctl.py`. A persistent
server (`server.py`) holds the robot connection; the arms keep torque and hold their pose
between your commands.

## Commands (run with the lerobot conda env python)

```bash
PY=~/miniconda3/envs/lerobot/bin/python
cd ~/lerobot/claude_robot

$PY ctl.py start                                 # connect robot (arms hold pose, torque ON)
$PY ctl.py state                                 # all joint positions
$PY ctl.py cam left_wrist                        # capture frame -> JPEG path (Read it to see)
$PY ctl.py move '{"left_gripper.pos": 50}'       # smooth 2 s move
$PY ctl.py move '{"right_shoulder_pan.pos": 10, "right_elbow_flex.pos": -15}' --duration 3
$PY ctl.py stop                                  # torque OFF + server exit
```

## Joint keys and ranges

`{left|right}_{shoulder_pan|shoulder_lift|elbow_flex|wrist_flex|wrist_roll}.pos` — normalized
**-100..100** (0 ≈ middle of that joint's calibrated range).
`{left|right}_gripper.pos` — **0..100** (0 = closed, 100 = open).
Left arm = orange, right arm = blue. Cameras: `left_wrist`, `right_wrist` (wrist-mounted, 640×480)
and `overhead` (Logitech C922, 1280×720, top-down scene view — use this one to locate objects;
the wrist cams suffer parallax at close range).

## Always-fresh overhead view

`~/lerobot/outputs/claude_robot/latest/overhead.jpg` is refreshed every few seconds by the
web app's scene watcher whenever the robot server is up. **Read it first for any spatial
task** — it's the current top-down scene, no ctl.py call needed. Orientation: the camera
faces the arms, so image-left = blue/RIGHT arm, image-right = orange/LEFT arm.

## Safety rules — follow these

1. **Look before you move**: capture both wrist cams and read `state` before the first move
   and after anything unexpected.
2. **Small steps**: change body joints by ≤ 20-30 units per move until you have visually
   confirmed the workspace is clear. The server also caps per-tick deltas.
3. **Gripper**: close gently — go to the value where the object is held, not 0. Holding a
   stalled gripper trips the motor's overload protection.
4. **Before `stop`**: torque releases and the arms DROOP. Move both arms to a low rest pose
   near the table first.
5. One bus owner at a time: the server refuses to start while `lerobot-teleoperate` or
   `lerobot-record` runs (`pkill -INT -f 'lerobot-(teleoperate|record)'` to free the buses).

## Troubleshooting

- Transient `ConnectionError: ... no status packet` on start → retry once, it's a dropped packet.
- Totally silent bus → check the power barrel jack on that arm (the orange follower's has been
  loose before).
- Server log: `~/lerobot/outputs/claude_robot/server.log`

## Pumpkin Bot persona (terminal)

`app/persona.md` is the kid-facing Halloween voice the web app's "pumpkin" agent uses.
A plain `claude` session here does NOT load it. To talk to Pumpkin Bot from the terminal:

```bash
./pumpkin.sh                 # interactive session with the persona appended
./pumpkin.sh -p "say hello"  # one-shot
```
