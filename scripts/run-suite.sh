#!/usr/bin/env bash
# run-suite.sh <tag> <model> <api_base> [workers] [batch]
# One 40-instance SWE-bench Pro run against a single OpenAI-compatible endpoint.
# Batched so the sweap-images never exceed free disk; images pruned after each batch.
set -u
TAG="${1:?tag}"; MODEL="${2:?model}"; APIBASE="${3:?api_base}"
W="${4:-4}"; BATCH="${5:-10}"
BASE="${BASE:-$HOME/swe-full731}"
OUT="$BASE/out-$TAG"
DS="$BASE/dataset/test.jsonl"
TOTAL=$(wc -l < "$DS")

# Per-run endpoint. mini-swe-agent/litellm read these from the environment, so each run
# targets its own server without touching the global ~/.config/mini-swe-agent/.env.
export OPENAI_API_BASE="$APIBASE"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-1234}"
export MSWEA_COST_TRACKING=ignore_errors

mkdir -p "$OUT"
echo "=== RUN $TAG | model=$MODEL | base=$APIBASE | workers=$W | n=$TOTAL | $(date +%F' '%T) ==="
for start in $(seq 0 "$BATCH" $((TOTAL-1))); do
  end=$((start+BATCH))
  python3 "$BASE/batch-images.py" "$DS" "$start" "$end" > "$BASE/batch-images.txt"
  echo "--- $TAG batch $start:$end pulling $(wc -l < "$BASE/batch-images.txt") images $(date +%T) ---"
  while read -r img; do docker pull "$img" >/dev/null 2>&1 || echo "PULL_FAIL $img"; done < "$BASE/batch-images.txt"
  df -h / | tail -1
  echo "--- $TAG batch $start:$end running $(date +%T) ---"
  ${MSWEA_BIN:-$HOME/swebench-pro/.venv/bin/mini-extra} swebench \
    -c swebench.yaml -c "$BASE/${OVERRIDE:-env-override.yaml}" \
    --subset "$BASE/dataset" --split test --slice "$start:$end" \
    --output "$OUT" --workers "$W" -m "$MODEL" 2>&1 | tail -20
  # prune this batch's images so the next batch has room
  while read -r img; do docker rmi "$img" >/dev/null 2>&1; done < "$BASE/batch-images.txt"
  docker ps -aq --filter "name=minisweagent-" | xargs -r docker rm -f >/dev/null 2>&1
  echo "BATCH_${TAG}_${start}_DONE $(date +%T)  trajectories=$(find "$OUT" -name '*.traj.json' | wc -l)"
done
echo "RUN_${TAG}_DONE $(date +%F' '%T)"
