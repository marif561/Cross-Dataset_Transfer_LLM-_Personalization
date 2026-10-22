#!/usr/bin/env bash
# Full USERPRINT pipeline: select -> cluster personas -> generate -> evaluate -> judge.
# Usage: bash scripts/run_pipeline.sh <lmsys_conversations.jsonl> <personachat_dialogues.jsonl> [out_dir]
# Models and judges come from configs/example.env (override with environment variables).
set -euo pipefail
CONV=${1:?"LMSYS conversations JSONL required"}
DIAL=${2:?"PersonaChat dialogues JSONL required"}
OUT=${3:-outputs}
source configs/example.env
mkdir -p "$OUT"

python -m userprint.select   --input "$CONV" --output "$OUT/selected.jsonl"
python -m userprint.personas --input "$OUT/selected.jsonl" --output "$OUT/personas.json" --embedder "$EMBEDDER"
python -m userprint.generate --dialogues "$DIAL" --personas "$OUT/personas.json" --models $GENERATORS \
                             --limit 300 --output "$OUT/responses.jsonl"
python -m userprint.evaluate --responses "$OUT/responses.jsonl" --out-dir "$OUT/eval"
python -m userprint.judge    --responses "$OUT/responses.jsonl" --dialogues "$DIAL" \
                             --personas "$OUT/personas.json" --judges $JUDGES --out-dir "$OUT/judge"
