#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?GPU id required}"
OUTPUT_ROOT="${2:?output root required}"
MIN_FREE_MIB="${AD_GENERATION_MIN_FREE_MIB:-18000}"

REPO=/mnt/sda/cbj/ImageGeneration/Self-main
AD_SOURCE=/mnt/sda/cbj/ImageGeneration/third_party/anomalydiffusion
AD_PYTHON=/mnt/sda/cbj/miniconda3/envs/Anomalydiffusion/bin/python
PREPARED="$REPO/results_for_gpt/plastic_bomo_chapter3_exp2_baseline_generation/prepared_v2_20261017_073039"
TRAINED="$REPO/results_for_gpt/plastic_bomo_chapter3_exp2_baseline_generation/ad_training_2000_gpu0_20261020014935/anomaly-checkpoints"
BASE_MODEL="$AD_SOURCE/models/ldm/text2img-large/model.ckpt"
BERT_SOURCE=/mnt/sda/cbj/GLIP-main/bert-base-uncased

finish() {
  code=$?
  echo "[已退出] AnomalyDiffusion 生成进程退出；退出码=$code"
}
trap finish EXIT

echo "[启动] AnomalyDiffusion 正式 A/B/C 生成；GPU=$GPU_ID；目标=240张。"
echo "[运行中] 等待 GPU $GPU_ID 可用显存达到 ${MIN_FREE_MIB} MiB。"
while true; do
  free_mib="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ' || true)"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
    break
  fi
  echo "[运行中] GPU $GPU_ID 当前空闲 ${free_mib:-unknown} MiB；驱动不可用或显存不足，30秒后重试。"
  sleep 30
done

if [[ ! -e "$AD_SOURCE/bert-base-uncased" ]]; then
  ln -s "$BERT_SOURCE" "$AD_SOURCE/bert-base-uncased"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PATH="$REPO/tools/exp2_ninja_shim:$PATH"
export PYTHONPATH="$REPO/tools:$REPO/../third_party/taming-transformers:$REPO/../third_party/openai-CLIP${PYTHONPATH:+:$PYTHONPATH}"

cd "$AD_SOURCE"
exec "$AD_PYTHON" -u "$REPO/tools/generate_plastic_bomo_exp2_ad.py" \
  --prepared "$PREPARED" \
  --ad-source "$AD_SOURCE" \
  --project-config "$TRAINED/configs/2026-10-20T02-00-28-project.yaml" \
  --base-model "$BASE_MODEL" \
  --spatial-encoder "$TRAINED/checkpoints/spatial_encoder.pt" \
  --embeddings "$TRAINED/checkpoints/embeddings.pt" \
  --training-checkpoint "$TRAINED/checkpoints/last.ckpt" \
  --output "$OUTPUT_ROOT" \
  --ddim-steps 200 \
  --ddim-eta 1.0
