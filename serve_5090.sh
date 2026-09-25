#!/usr/bin/env bash
# Start the pi0-FAST policy server on the 5090 (gpu5090) with an SSH tunnel back here.
# Run this in its own terminal and leave it open; Ctrl-C stops server + tunnel.
# The model itself is chosen by the CLIENT (biarm_pi0fast_client.sh) at handshake.
exec ssh -L 8080:localhost:8080 gpu5090 \
  "cd ~/pi-serve && exec .venv/bin/python -m lerobot.async_inference.policy_server --host=localhost --port=8080"
