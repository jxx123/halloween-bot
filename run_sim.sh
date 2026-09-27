#!/usr/bin/env bash
# Bring up the sim stack on the 5090: π₀-FAST policy server (:8081, ~/pi-serve venv with its
# pi0_fast shims — never reinstall that venv), MuJoCo sim server (:8399), web app (:8500).
# The :8081 policy server is SHARED: the 1080's real robot reaches it through its SSH tunnel
# (workstation:8080 -> here:8081; two pi0fast copies don't fit on the GPU). One π run at a time:
# the sim refuses to start, and yields mid-run, while another client is connected.
# Logs: ~/lerobot/outputs/claude_robot/*.log
set -euo pipefail
cd "$(dirname "$0")"
LOG=~/lerobot/outputs/claude_robot
mkdir -p "$LOG"
listening() { ss -ltnH "sport = :$1" | grep -q .; }  # by port: pgrep -f also matches shells quoting the cmd
if ! listening 8081; then
  (cd ~/pi-serve && setsid .venv/bin/python -m lerobot.async_inference.policy_server \
      --host=127.0.0.1 --port=8081 >>"$LOG/policy_server_sim.log" 2>&1 &)
  echo "policy server starting on :8081 (the first π run loads the model, ~50 s)"
fi
.venv/bin/python -m halloween_bot.ctl start --sim --scene "${SIM_SCENE:-toys}"  # SIM_SCENE=candy ./run_sim.sh
if ! listening 8500; then
  setsid .venv/bin/python -m halloween_bot.app.webapp >>"$LOG/webapp.log" 2>&1 &
  sleep 2
fi
echo "dashboard: http://127.0.0.1:8500/   face: http://127.0.0.1:8500/face?theme=halloween"
