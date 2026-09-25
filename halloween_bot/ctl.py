#!/usr/bin/env python
"""CLI client for the SO101 control server — the tool a Claude Code agent calls via Bash.

  python ctl.py start                                # launch server (arms hold pose, torque ON)
  python ctl.py state                                # joint positions, normalized
  python ctl.py move '{"left_gripper.pos": 50}'      # smooth move, default 2 s
  python ctl.py move '{"right_elbow_flex.pos": -20, "right_wrist_flex.pos": 10}' --duration 3
  python ctl.py cam left_wrist                       # capture frame -> prints JPEG path
  python ctl.py cam right_wrist
  python ctl.py stop                                 # torque OFF (support the arms!) + server exit
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


def cmd_start(_):
    if server_alive():
        print("server already running")
        return
    Path(LOG).parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "ab") as log:
        subprocess.Popen(["setsid", PYTHON, SERVER], stdout=log, stderr=log, start_new_session=True)
    for _ in range(40):  # connect takes a few seconds
        time.sleep(1)
        if server_alive():
            print("server up, robot connected (torque ON)")
            return
    sys.exit(f"server did not come up — check log: {LOG}")


def cmd_state(_):
    print(json.dumps(call("/state"), indent=2))


def cmd_move(args):
    targets = json.loads(args.targets)
    print(json.dumps(call("/move", {"targets": targets, "duration": args.duration}), indent=2))


def cmd_cam(args):
    print(json.dumps(call(f"/cam?name={args.name}"), indent=2))


def cmd_stop(_):
    print(json.dumps(call("/stop", {})))


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = p.add_subparsers(dest="cmd", required=True)
sub.add_parser("start").set_defaults(fn=cmd_start)
sub.add_parser("state").set_defaults(fn=cmd_state)
mp = sub.add_parser("move")
mp.add_argument("targets", help='JSON dict, e.g. \'{"left_gripper.pos": 50}\'')
mp.add_argument("--duration", type=float, default=2.0)
mp.set_defaults(fn=cmd_move)
cp = sub.add_parser("cam")
cp.add_argument("name", choices=["left_wrist", "right_wrist"])
cp.set_defaults(fn=cmd_cam)
sub.add_parser("stop").set_defaults(fn=cmd_stop)

args = p.parse_args()
args.fn(args)
