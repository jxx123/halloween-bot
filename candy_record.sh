#!/usr/bin/env bash
# Record real teleop candy demos that CO-TRAIN with the 5090 sim data.
#
# Schema locked to the sim (sim_candy_v0): 3 cameras named overhead/left_wrist/right_wrist,
# observation.state + action 12-d in left-then-right order, 30 fps. The base_0_rgb /
# left_wrist_0_rgb / right_wrist_0_rgb renames happen at policy time, so keep the rig names here.
#
# Usage:
#   ./candy_record.sh                                  # 1 pick_place demo, chocolate bar
#   TASK=pick_place N=20 ./candy_record.sh          # any-candy: grab whichever is convenient
#   TASK=give_human N=20 ./candy_record.sh
#   TASK=handover   N=10 ./candy_record.sh          # the hard one — most valuable
#
# REQUIRES the halloween_bot robot server STOPPED first (it owns the buses + cameras):
#   pkill -f 'halloween_bot.serve[r]'
# and the leader arms plugged in. Restart the server after with: python -m halloween_bot.ctl start
#
# Keyboard during recording (needs the desktop): right arrow = end episode early,
# left arrow = re-record, ESC = stop.
set -euo pipefail

PY=~/miniconda3/envs/lerobot-eval/bin
TASK="${TASK:-pick_place}"
CANDY="${CANDY:-chocolate bar}"
N="${N:-1}"
EPISODE_TIME="${EPISODE_TIME:-30}"
RESET_TIME="${RESET_TIME:-10}"

case "$TASK" in
  pick_place) SENTENCE="Put a candy on the plate." ;;
  give_human) SENTENCE="Give a candy to the person." ;;
  handover)   SENTENCE="Pick up a candy, hand it to the other arm, and put it on the plate." ;;
  *) echo "TASK must be pick_place | give_human | handover" >&2; exit 2 ;;
esac

SLUG=$(echo "${CANDY}" | tr ' A-Z' '_a-z')
ROOT="$HOME/lerobot/outputs/datasets/candy_${TASK}_${SLUG}"
echo "Recording $N x ${EPISODE_TIME}s  |  task=$TASK  candy='$CANDY'"
echo "  \"$SENTENCE\""
echo "  -> $ROOT"

# by-path / by-id device paths mirror halloween_bot/server.py
L_FOLLOWER=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6057297-if00
R_FOLLOWER=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083852-if00
L_LEADER=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6055292-if00
R_LEADER=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6055274-if00
OVERHEAD=/dev/v4l/by-id/usb-046d_C922_Pro_Stream_Webcam_59D955BF-video-index0
LEFT_WRIST=/dev/v4l/by-path/pci-0000:00:14.0-usb-0:11:1.0-video-index0    # orange
RIGHT_WRIST=/dev/v4l/by-path/pci-0000:00:14.0-usb-0:12:1.0-video-index0   # blue

exec "$PY/lerobot-record" \
  --robot.type=bi_so_follower \
  --robot.id=bimanual \
  --robot.left_arm_config.port="$L_FOLLOWER" \
  --robot.left_arm_config.max_relative_target=null \
  --robot.right_arm_config.port="$R_FOLLOWER" \
  --robot.right_arm_config.max_relative_target=null \
  --robot.cameras="{
      overhead:   {type: opencv, index_or_path: $OVERHEAD,   width: 640, height: 480, fps: 30, fourcc: MJPG},
      left_wrist: {type: opencv, index_or_path: $LEFT_WRIST, width: 640, height: 480, fps: 30, fourcc: MJPG},
      right_wrist:{type: opencv, index_or_path: $RIGHT_WRIST,width: 640, height: 480, fps: 30, fourcc: MJPG}
    }" \
  --teleop.type=bi_so_leader \
  --teleop.id=bimanual \
  --teleop.left_arm_config.port="$L_LEADER" \
  --teleop.right_arm_config.port="$R_LEADER" \
  --dataset.repo_id="jxx123/candy_${TASK}_${SLUG}" \
  --dataset.root="$ROOT" \
  --dataset.single_task="$SENTENCE" \
  --dataset.fps=30 \
  --dataset.num_episodes="$N" \
  --dataset.episode_time_s="$EPISODE_TIME" \
  --dataset.reset_time_s="$RESET_TIME" \
  --dataset.push_to_hub=false \
  --display_data=false
