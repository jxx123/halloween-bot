#!/usr/bin/env python
"""CLI client for the SO101 control server — the tool a Claude Code agent calls via Bash.

  python ctl.py start                                # launch server (arms hold pose, torque ON)
  python ctl.py state                                # joint positions, normalized
  python ctl.py move '{"left_gripper.pos": 50}'      # smooth move, default 2 s
  python ctl.py move '{"right_elbow_flex.pos": -20, "right_wrist_flex.pos": 10}' --duration 3
  python ctl.py cam left_wrist                       # capture frame -> prints JPEG path
  python ctl.py cam right_wrist                      # (also: overhead, and scene in sim)
  python ctl.py stop                                 # torque OFF (support the arms!) + server exit

π₀-FAST policy (pick a toy, put it in the basket) — works on the sim and, once wired, the real rig:
  python ctl.py policy                               # default task, 30 s, blocks + prints progress
  python ctl.py policy "Grasp the toy and place it in the basket." --seconds 20
  python ctl.py policy-stop

MuJoCo twin (5090):
  python ctl.py start --sim                          # sim server on the same port/API
  python ctl.py sim-reset [--randomize]              # arms to rest pose, toys re-placed
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PORT = 8399
BASE = f"http://127.0.0.1:{PORT}"
PYTHON = sys.executable
SERVER = str(Path(__file__).parent / "server.py")
LOG = str(Path.home() / "lerobot/outputs/claude_robot/server.log")
SIM_LOG = str(Path.home() / "lerobot/outputs/claude_robot/sim_server.log")
DEFAULT_TASK = "Grasp the toy and place it in the basket."


def call(path, payload=None, timeout=120):
    req = urllib.request.Request(BASE + path)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"server error {e.code}: {e.read().decode()}")


def server_alive():
    try:
        call("/state", timeout=5)
        return True
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return False


def cmd_start(args):
    if server_alive():
        print("server already running")
        return
    Path(LOG).parent.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, "-m", "halloween_bot.sim.server"] if args.sim else [PYTHON, SERVER]
    log_path = SIM_LOG if args.sim else LOG
    with open(log_path, "ab") as log:
        subprocess.Popen(["setsid", *cmd], stdout=log, stderr=log, start_new_session=True)
    for _ in range(60):  # robot connect / sim model build + EGL take a few seconds
        time.sleep(1)
        if server_alive():
            print("SIM server up (MuJoCo twin, same API)" if args.sim else "server up, robot connected (torque ON)")
            return
    sys.exit(f"server did not come up — check log: {log_path}")


def cmd_state(_):
    print(json.dumps(call("/state"), indent=2))


def cmd_move(args):
    targets = json.loads(args.targets)
    print(json.dumps(call("/move", {"targets": targets, "duration": args.duration}), indent=2))


def cmd_cam(args):
    print(json.dumps(call(f"/cam?name={args.name}"), indent=2))


def cmd_stop(_):
    print(json.dumps(call("/stop", {})))


def cmd_policy(args):
    res = call("/policy", {"task": args.task, "seconds": args.seconds, "lockstep": args.lockstep})
    if not res.get("ok"):
        sys.exit(f"policy start failed: {res.get('error')}")
    print(f"π policy started: task={args.task!r} seconds={args.seconds} lockstep={args.lockstep}", flush=True)
    if args.no_wait:
        return
    last = None
    while True:
        time.sleep(2)
        st = call("/policy")
        line = f"[{st['phase']}] t={st['elapsed_s']}s executed={st['executed']} chunks={st['chunks']}"
        if line != last:
            print(line, flush=True)
            last = line
        if not st["running"]:
            break
    print(json.dumps(st, indent=2))


def cmd_policy_stop(_):
    print(json.dumps(call("/policy/stop", {}), indent=2))


def cmd_sim_reset(args):
    print(json.dumps(call("/sim/reset", {"randomize": args.randomize}), indent=2))


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(dest="cmd", required=True)
sp = sub.add_parser("start")
sp.add_argument("--sim", action="store_true", help="start the MuJoCo twin instead of the real robot")
sp.set_defaults(fn=cmd_start)
sub.add_parser("state").set_defaults(fn=cmd_state)
mp = sub.add_parser("move")
mp.add_argument("targets", help='JSON dict, e.g. \'{"left_gripper.pos": 50}\'')
mp.add_argument("--duration", type=float, default=2.0)
mp.set_defaults(fn=cmd_move)
cp = sub.add_parser("cam")
cp.add_argument("name", choices=["left_wrist", "right_wrist", "overhead", "scene"])
cp.set_defaults(fn=cmd_cam)
sub.add_parser("stop").set_defaults(fn=cmd_stop)
pp = sub.add_parser("policy", help="run the π₀-FAST policy (blocks until it finishes)")
pp.add_argument("task", nargs="?", default=DEFAULT_TASK)
pp.add_argument("--seconds", type=float, default=30.0)
pp.add_argument("--lockstep", action="store_true", help="sim only: pause physics during inference")
pp.add_argument("--no-wait", action="store_true", help="return right after starting")
pp.set_defaults(fn=cmd_policy)
sub.add_parser("policy-stop").set_defaults(fn=cmd_policy_stop)
rp = sub.add_parser("sim-reset")
rp.add_argument("--randomize", action="store_true")
rp.set_defaults(fn=cmd_sim_reset)

args = p.parse_args()
args.fn(args)
