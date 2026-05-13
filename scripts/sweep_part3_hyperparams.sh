#!/usr/bin/env bash
# Part3 C_hybrid 超参细调：对每条序列跑多组配置，再 eval 汇总（JSON 含 aggregation 排名）。
# 默认只扫 calendar（快）；设置 SWEEP_SEQUENCES="calendar city foliage walk" 可扫全 VID4；
# 或直接用 scripts/batch_part3_vid4_sweep.sh。
# 聚合权重（可选）：AGG_FRAME_WEIGHT / AGG_TEMPORAL_WEIGHT（默认 0.45 / 0.55，偏视频时序）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
SEQ_LIST="${SWEEP_SEQUENCES:-calendar}"

# 配置名 | fusion_gen_scale | temporal_blend | rule_zeta | 额外 inference 参数
read -r -d '' CONFIGS <<'EOF' || true
v2base|1.0|0.24|2.5|
f055|0.55|0.24|2.5|
f060|0.60|0.24|2.5|
lpips_soft|1.0|0.20|2.0|
strong_time|1.0|0.30|2.5|
combo|0.58|0.26|2.5|
EOF

run_one_seq() {
  local seq="$1"
  local name="$2"
  local fgs="$3"
  local tbl="$4"
  local zta="$5"
  local extra="$6"
  local out_dir="results/part3_${seq}_C_hybrid_${name}"
  echo "=== [$seq] $name -> $out_dir ==="
  cmd=(
    "$PY" scripts/inference_part3.py --mode C_hybrid
    --input_dir "data/BIx4/$seq"
    --output_dir "$out_dir"
    --device "$DEVICE"
    --fusion_gen_scale "$fgs"
    --temporal_blend "$tbl"
    --rule_zeta "$zta"
  )
  if [[ -n "$extra" ]]; then
    # shellcheck disable=SC2206
    cmd+=($extra)
  fi
  "${cmd[@]}"
}

for seq in $SEQ_LIST; do
  if [[ ! -d "data/BIx4/$seq" ]]; then
    echo "[skip] missing data/BIx4/$seq" >&2
    continue
  fi
  while IFS='|' read -r name fgs tbl zta extra; do
    [[ -z "$name" ]] && continue
    [[ "$name" =~ ^# ]] && continue
    run_one_seq "$seq" "$name" "$fgs" "$tbl" "$zta" "$extra"
  done <<<"$CONFIGS"
done

# 汇总 eval：methods = part3_{seq}_C_hybrid_<cfg> 对每个 cfg
METHODS=()
while IFS='|' read -r name _rest; do
  [[ -z "$name" ]] && continue
  [[ "$name" =~ ^# ]] && continue
  METHODS+=("part3_{seq}_C_hybrid_${name}")
done <<<"$CONFIGS"

OUT_JSON="${SWEEP_EVAL_JSON:-results/eval_part3_sweep_hyperparams.json}"
echo "=== eval -> $OUT_JSON ==="
AGG_FW="${AGG_FRAME_WEIGHT:-0.45}"
AGG_TW="${AGG_TEMPORAL_WEIGHT:-0.55}"

"$PY" scripts/eval_pipeline.py \
  --gt_root data/GT \
  --results_root results \
  --methods "${METHODS[@]}" \
  --sequences $SEQ_LIST \
  --crop_border 4 \
  --skip_fid \
  --agg_frame_weight "$AGG_FW" \
  --agg_temporal_weight "$AGG_TW" \
  --save_json "$OUT_JSON" \
  --device "$DEVICE"

echo "[done] see $OUT_JSON (summary + per_sequence)"
