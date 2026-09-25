# halloween-bot 🎃🤖

A trick-or-treat robot built on two [SO101 arms](https://github.com/TheRobotStudio/SO-ARM100):
a glowing pumpkin face that talks to kids, always-on ears, motion-triggered greetings,
and a Claude agent brain that sees through three cameras and moves both arms.

Built on [LeRobot](https://github.com/huggingface/lerobot) (`bi_so_follower` / `bi_so_leader`).

## Architecture

```
Browser ──► webapp :8500 ─── dashboard (/), robot face (/face), SSE events, chat
   │            │
   │ MJPEG      │ instructions → agent layer (abstract) → claude | pumpkin persona | codex | echo
   ▼            ▼
robot server :8399 (holds arms + 3 cams, torque on; /move /state /cam /stream /teleop)
   │
   ▼
bimanual SO101 (2 leaders + 2 followers, Feetech STS3215)
```

- **`halloween_bot/server.py`** — persistent robot daemon: smooth interpolated `/move`
  with per-tick safety caps, MJPEG `/stream`, built-in leader→follower `/teleop` (50 Hz,
  cameras stay live), camera self-recovery.
- **`halloween_bot/ctl.py`** — thin CLI (`start|state|move|cam|stop`) that agents drive via shell.
- **`halloween_bot/app/`** — web app: dashboard, full-screen animated face
  (`/face?theme=halloween&listen=1&agent=pumpkin`), Cartesia TTS proxy, scene watcher
  (fresh overhead frame every second + motion-triggered agent wake-ups), master ▶ start button.
- **`halloween_bot/app/agents.py`** — pluggable agent backends; the `pumpkin` persona
  (`app/persona.md`) speaks only kid-friendly lines through the voice.
- **`serve_5090.sh` / `biarm_pi0fast_client.sh`** — remote policy serving: π₀-FAST bimanual
  checkpoint on a 5090 box, robot client here over an SSH-tunneled gRPC link.

## Setup

```bash
uv venv --python 3.12
uv pip install -e .
cp halloween_bot/app/tts_config.example.json halloween_bot/app/tts_config.json  # add your Cartesia key
```

Hardware bring-up (ports, calibration ids, udev rules) follows the standard LeRobot SO101
docs; this repo expects calibration ids `bimanual_left` / `bimanual_right` and stable
`/dev/serial/by-id/...` port paths (edit the constants at the top of `server.py` for your rig).

## Run

```bash
.venv/bin/python -m halloween_bot.ctl start          # robot server (arms hold pose)
.venv/bin/python -m halloween_bot.app.webapp &       # web app on :8500
google-chrome --kiosk "http://127.0.0.1:8500/face?theme=halloween&listen=1&agent=pumpkin"
```

Press **▶ start** on the dashboard: ears on, motion reactions on, Pumpkin Bot hosts the porch.
