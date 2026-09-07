#!/usr/bin/env bash
# Copy the FP8 checkpoint spark-1 -> spark-2 over the 200GbE fabric, with a cache warden
# on the receiving side (a 176 GB write fills page cache and that is what wedges GB10).
# Verification compares the SET of content hashes, order-independent, and treats a zero
# or mismatched file count as fatal. Sources are never touched.
set -u
SRC=/data/models/qwen3.8-flash-next-fp8
DST=/data/models/qwen3.8-flash-next-fp8
FABRIC_SRC=${NODE0_FABRIC}          # spark-1 over fabric

mkdir -p "$DST"
setsid nohup python3 /tmp/cache-warden.py "$DST" 20 > "$HOME/logs/cache-warden-copy.log" 2>&1 < /dev/null &
WARDEN=$!
echo "=== copying over fabric $(date +%H:%M:%S) ==="
rsync -a --info=progress2 --no-inc-recursive \
      -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
      "user@${FABRIC_SRC}:${SRC}/" "$DST/" 2>&1 | tail -3

echo "=== hashing source $(date +%H:%M:%S) ==="
ssh -o BatchMode=yes "user@${FABRIC_SRC}" "cd $SRC && find . -type f | xargs -r sha256sum | awk '{print \$1}' | sort" > /tmp/fp8.src.sha
echo "=== hashing destination $(date +%H:%M:%S) ==="
( cd "$DST" && find . -type f | xargs -r sha256sum | awk '{print $1}' | sort ) > /tmp/fp8.dst.sha

s=$(wc -l < /tmp/fp8.src.sha); d=$(wc -l < /tmp/fp8.dst.sha)
kill $WARDEN 2>/dev/null
if [ "$s" -eq 0 ] || [ "$d" -eq 0 ]; then
  echo "VERIFY_FAIL fp8: zero files hashed (src=$s dst=$d) -- transfer did not run"
elif [ "$s" -ne "$d" ]; then
  echo "VERIFY_FAIL fp8: file count mismatch src=$s dst=$d"
elif cmp -s /tmp/fp8.src.sha /tmp/fp8.dst.sha; then
  echo "VERIFY_OK fp8: $s files, all content hashes match"
else
  echo "VERIFY_FAIL fp8: $s files but content hashes differ"
  diff /tmp/fp8.src.sha /tmp/fp8.dst.sha | head -5
fi
echo "FP8_COPY_DONE $(date +%F' '%T)  free=$(df -h /data | awk 'NR==2{print $4}')"
