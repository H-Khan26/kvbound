#!/usr/bin/env bash
# Pins the GPU clock so benchmark runs are comparable. Native Linux only; needs root.
#   sudo scripts/lock_clocks.sh [MHz]    lock (default: the card's maximum SM clock)
#   sudo scripts/lock_clocks.sh reset    return to driver-managed clocks
set -euo pipefail

if [[ "${1:-}" == "reset" ]]; then
  nvidia-smi --reset-gpu-clocks
  exit 0
fi

max_mhz=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits | head -n1)
mhz="${1:-$max_mhz}"

nvidia-smi --persistence-mode=1
nvidia-smi --lock-gpu-clocks="${mhz},${mhz}"
echo "locked GPU clock at ${mhz} MHz; verify with: make doctor"
