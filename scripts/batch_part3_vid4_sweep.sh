#!/usr/bin/env bash
# 全 VID4 批量跑 Part3 超参扫参（6 组配置 × 4 序列），并写出带 aggregation 的 eval JSON。
# 耗时很长；仅冒烟可： SWEEP_SEQUENCES=calendar bash scripts/batch_part3_vid4_sweep.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SWEEP_SEQUENCES="${SWEEP_SEQUENCES:-calendar city foliage walk}"
export SWEEP_EVAL_JSON="${SWEEP_EVAL_JSON:-results/eval_part3_sweep_vid4_full.json}"

exec bash scripts/sweep_part3_hyperparams.sh
