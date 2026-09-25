#!/usr/bin/env python
"""Scripted pick-and-place: put the plush toys into the basket with the right (blue) arm.

Direct bus control at 50 Hz. Waypoints were measured in-session by visual servoing.
Fixes over the interactive attempts:
  - grip force is MAINTAINED during transport (keep commanding below the reached
    position) instead of being silently released
  - carry with claws tilted up so gravity pushes the toy into the claw vee
  - release INSIDE the basket mouth (zero drop height, can't bounce out)

Checkpoints frames are saved to outputs/claude_robot/frames/ck_*.jpg for review.

Usage:  pick_toys.py [--skip chick] [--skip monkey]
"""
import argparse
import time
from pathlib import Path

import cv2

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

FRAMES = Path.home() / "lerobot/outputs/claude_robot/frames"
HZ = 50
ROLL = -10  # wrist_roll held constant; grasp poses were measured at this roll

# ---- waypoints: {pan, lift, elbow, wf(=wrist_flex)} -------------------------
CHICK_HOVER = dict(pan=-26, lift=-71, elbow=86, wf=45)
CHICK_GRASP = dict(pan=-26, lift=-82, elbow=93, wf=45)
MONKEY_HOVER = dict(pan=-38, lift=-71, elbow=86, wf=45)
MONKEY_GRASP = dict(pan=-38, lift=-82, elbow=93, wf=45)
CRADLE = dict(pan=None, lift=-58, elbow=62, wf=-15)  # pan=None -> keep current
SWING = dict(pan=-43, lift=-50, elbow=25, wf=-15)
TIP = dict(pan=-43, lift=-56, elbow=24, wf=30)
INTO = dict(pan=-43, lift=-63, elbow=26, wf=30)
RISE = dict(pan=-43, lift=-45, elbow=30, wf=30)
SURVEY = dict(pan=-22, lift=-63, elbow=80, wf=40)
PARK = dict(pan=-5, lift=-86, elbow=95, wf=45)

OPEN, CLOSED = 90.0, 0.0
HOLD_MARGIN = 4.0  # keep commanding this far below reached -> sustained grip force
EMPTY_THRESH = 4.0  # gripper closing below this means the claws met: nothing grasped


def make_robot() -> BiSOFollower:
    return BiSOFollower(
        BiSOFollowerConfig(
            id="bimanual",
            left_arm_config=SOFollowerConfig(
                port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057297-if00",
                max_relative_target=20.0,
            ),
            right_arm_config=SOFollowerConfig(
                port="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083852-if00",
                max_relative_target=20.0,
            ),
            cameras={"right_wrist": OpenCVCameraConfig(index_or_path="/dev/video0", fps=30, width=640, height=480, fourcc="MJPG")},
        )
    )


class Arm:
    """Right-arm waypoint runner; left arm is frozen at its startup pose."""

    def __init__(self, robot: BiSOFollower):
        self.robot = robot
        self.grip_cmd = None  # gripper value commanded every tick (None until first set)
        left = robot.left_arm.bus.sync_read("Present_Position")
        self.left_hold = {f"left_{m}.pos": v for m, v in left.items()}

    def read(self) -> dict:
        right = self.robot.right_arm.bus.sync_read("Present_Position")
        return {f"right_{m}.pos": v for m, v in right.items()}

    def act(self, right_targets: dict):
        self.robot.send_action({**self.left_hold, **right_targets})

    def goto(self, wp: dict, duration: float, grip: float | None = None):
        """Interpolate right-arm joints to waypoint. grip overrides the held gripper command."""
        if grip is not None:
            self.grip_cmd = grip
        cur = self.read()
        tgt = dict(cur)
        mapping = dict(pan="right_shoulder_pan.pos", lift="right_shoulder_lift.pos",
                       elbow="right_elbow_flex.pos", wf="right_wrist_flex.pos")
        for k, joint in mapping.items():
            if wp.get(k) is not None:
                tgt[joint] = float(wp[k])
        tgt["right_wrist_roll.pos"] = ROLL
        if self.grip_cmd is not None:
            tgt["right_gripper.pos"] = self.grip_cmd
        steps = max(1, int(duration * HZ))
        for i in range(1, steps + 1):
            a = i / steps
            action = {k: cur[k] + (tgt[k] - cur[k]) * a for k in tgt}
            if self.grip_cmd is not None:
                action["right_gripper.pos"] = self.grip_cmd  # constant force, not interpolated
            self.act(action)
            time.sleep(1.0 / HZ)

    def set_grip(self, value: float, duration: float = 1.5) -> float:
        """Move gripper only; returns reached position."""
        cur = self.read()["right_gripper.pos"]
        steps = max(1, int(duration * HZ))
        for i in range(1, steps + 1):
            g = cur + (value - cur) * i / steps
            self.act({"right_gripper.pos": g})
            time.sleep(1.0 / HZ)
        self.grip_cmd = value
        time.sleep(0.3)
        return self.read()["right_gripper.pos"]

    def snap(self, name: str) -> str:
        frame = self.robot.left_arm.cameras["right_wrist"].async_read(timeout_ms=2000)
        FRAMES.mkdir(parents=True, exist_ok=True)
        path = FRAMES / f"ck_{name}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"  [frame] {path}", flush=True)
        return str(path)


def deliver(arm: Arm, toy: str, hover: dict, grasp: dict) -> bool:
    print(f"--- {toy}: approach", flush=True)
    arm.goto(hover, 3.0, grip=OPEN)
    arm.goto(grasp, 2.0)
    arm.snap(f"{toy}_1_pregrasp")

    print(f"--- {toy}: close gripper", flush=True)
    reached = arm.set_grip(CLOSED, 1.5)
    print(f"  gripper reached {reached:.1f}", flush=True)
    if reached < EMPTY_THRESH:
        print(f"  !! EMPTY GRASP (< {EMPTY_THRESH}) — aborting {toy}, review pregrasp frame", flush=True)
        arm.set_grip(OPEN, 1.0)
        arm.goto(SURVEY, 2.5)
        return False
    hold = max(reached - HOLD_MARGIN, 0.0)
    arm.grip_cmd = hold
    print(f"  holding with continuous command {hold:.1f}", flush=True)

    print(f"--- {toy}: cradle + swing to basket", flush=True)
    arm.goto(CRADLE, 3.0)
    arm.snap(f"{toy}_2_cradle")
    arm.goto(SWING, 4.0)
    arm.snap(f"{toy}_3_over_basket")

    print(f"--- {toy}: lower into basket and release", flush=True)
    arm.goto(TIP, 2.0)
    arm.goto(INTO, 2.0)
    arm.set_grip(OPEN, 1.0)
    arm.goto(RISE, 1.5)
    arm.goto(SURVEY, 2.5)
    arm.snap(f"{toy}_4_after")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", action="append", default=[], choices=["chick", "monkey"])
    ap.add_argument("--probe", action="store_true", help="hover+snap both grasp poses, no gripping")
    args = ap.parse_args()

    robot = make_robot()
    robot.connect(calibrate=False)
    print("connected", flush=True)
    arm = Arm(robot)
    try:
        if args.probe:
            for toy, hover, grasp in [("chick", CHICK_HOVER, CHICK_GRASP), ("monkey", MONKEY_HOVER, MONKEY_GRASP)]:
                if toy in args.skip:
                    continue
                arm.goto(hover, 3.0, grip=OPEN)
                arm.goto(grasp, 2.0)
                arm.snap(f"{toy}_probe")
            arm.goto(PARK, 3.0, grip=40.0)
            return
        results = {}
        if "chick" not in args.skip:
            results["chick"] = deliver(arm, "chick", CHICK_HOVER, CHICK_GRASP)
        if "monkey" not in args.skip:
            results["monkey"] = deliver(arm, "monkey", MONKEY_HOVER, MONKEY_GRASP)
        print("RESULTS:", results, flush=True)
        arm.goto(PARK, 3.0, grip=40.0)
    finally:
        robot.disconnect()
        print("disconnected (torque off)", flush=True)


if __name__ == "__main__":
    main()
