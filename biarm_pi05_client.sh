#!/usr/bin/env bash
# Real-arm eval of a fine-tuned pi0.5 candy checkpoint served from the 5090 (:8082).
#
# SAFETY: keeps max_relative_target=20 per tick on both arms — never loosen for an
# undertrained checkpoint. Contact caps + hardware range limits are the containment.
#
# Prereqs, in order:
#   1. sim numbers justify it AND Jinyu says go in the 1080 session.
#   2. candy + plate on the table, workspace otherwise clear, lights on, human watching.
#   3. stop the halloween_bot server (it owns the buses + cameras):
#        pkill -f 'halloween_bot.serve[r]'
#   4. tunnel the 5090 pi0.5 policy server:
#        ssh -f -N -L 8082:localhost:8082 gpu5090
#
# Usage:  CKPT=<path-or-repo the 5090 server resolves> TASK="Pick up the chocolate bar and put it on the plate." ./biarm_pi05_client.sh
set -euo pipefail

PY=~/miniconda3/envs/lerobot-eval/bin
CKPT="${CKPT:?set CKPT to the merged pi0.5 checkpoint id the :8082 server loads}"
TASK="${TASK:-Pick up the candy and put it on the plate.}"

L_FOLLOWER=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057297-if00
R_FOLLOWER=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083852-if00
OVERHEAD=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_59D955BF-video-index0
LEFT_WRIST=/dev/v4l/by-path/pci-0000:00:14.0-usb-0:11:1.0-video-index0    # orange
RIGHT_WRIST=/dev/v4l/by-path/pci-0000:00:14.0-usb-0:12:1.0-video-index0   # blue

exec "$PY/python" -m lerobot.async_inference.robot_client \
  --robot.type=bi_so_follower \
  --robot.id=bimanual \
  --robot.left_arm_config.port="$L_FOLLOWER" \
  --robot.left_arm_config.max_relative_target=20.0 \
  --robot.left_arm_config.use_degrees=false \
  --robot.right_arm_config.port="$R_FOLLOWER" \
  --robot.right_arm_config.max_relative_target=20.0 \
  --robot.right_arm_config.use_degrees=false \
  --robot.cameras="{
      overhead:    {type: opencv, index_or_path: $OVERHEAD,    width: 640, height: 480, fps: 30, fourcc: MJPG},
      left_wrist:  {type: opencv, index_or_path: $LEFT_WRIST,  width: 640, height: 480, fps: 30, fourcc: MJPG},
      right_wrist: {type: opencv, index_or_path: $RIGHT_WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG}
    }" \
  --policy_type=pi05 \
  --pretrained_name_or_path="$CKPT" \
  --policy_device=cuda \
  --server_address=localhost:8082 \
  --task="$TASK" \
  --actions_per_chunk=50 \
  --fps=30
