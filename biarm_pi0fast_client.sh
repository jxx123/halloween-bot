#!/usr/bin/env bash
# Drive the bimanual SO101 with pi0-FAST served from the 5090 (start serve_5090.sh first).
#
# Requires the local robot server to be OFF (it owns the buses):
#   ~/miniconda3/envs/lerobot/bin/python ~/lerobot/claude_robot/ctl.py stop
#
# Cameras are named to match the policy's training features exactly:
#   base_0_rgb = overhead C922, left_wrist_0_rgb = orange wrist, right_wrist_0_rgb = blue wrist
set -euo pipefail

TASK="${TASK:-Grasp the toy and place it in the basket.}"

exec ~/miniconda3/envs/lerobot-eval/bin/python -m lerobot.async_inference.robot_client \
  --robot.type=bi_so_follower \
  --robot.id=bimanual \
  --robot.left_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057297-if00 \
  --robot.left_arm_config.max_relative_target=20.0 \
  --robot.right_arm_config.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083852-if00 \
  --robot.right_arm_config.max_relative_target=20.0 \
  --robot.cameras="{
      base_0_rgb:       {type: opencv, index_or_path: /dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_59D955BF-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
      left_wrist_0_rgb: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG},
      right_wrist_0_rgb: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30, fourcc: MJPG}
    }" \
  --policy_type=pi0_fast \
  --pretrained_name_or_path=delvingdeep/pi0fast-so101-bimanual \
  --policy_device=cuda \
  --server_address=localhost:8080 \
  --task="$TASK" \
  --fps=30
