#!/usr/bin/env bash
# GB10 pre-launch ritual — run on BOTH nodes after EVERY reboot, before launching vLLM.
# (tonyd2wild GLM-5.3-Flash repo: vm.swappiness=0 does NOT survive reboot; with swap
# active the kernel pages vLLM out mid-load and triggers a UVM driver livelock that
# does not recover on its own. drop_caches frees page cache from downloads/reads.)
set -euo pipefail
sudo bash -c '
  echo vm.swappiness = 0 > /etc/sysctl.d/99-llm-serving.conf
  sysctl -p /etc/sysctl.d/99-llm-serving.conf >/dev/null
  sync
  echo 3 > /proc/sys/vm/drop_caches
  swapoff -a 2>/dev/null; swapon -a 2>/dev/null || true
' 2>/dev/null
echo "swappiness=$(cat /proc/sys/vm/swappiness)"
free -g | head -2