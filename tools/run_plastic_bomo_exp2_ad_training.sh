#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:?GPU id required}"
MAX_STEPS="${2:?max optimizer steps required}"
RUN_ROOT="${3:?run root required}"
# The 10-step 512x512 optimizer pilot peaked at 22,284 MiB on an RTX 3090.
# Require enough headroom for allocator and driver variation before loading.
MIN_FREE_MIB="${AD_MIN_FREE_MIB:-23500}"

REPO=/mnt/sda/cbj/ImageGeneration/Self-main
AD_SOURCE=/mnt/sda/cbj/ImageGeneration/third_party/anomalydiffusion
AD_PYTHON=/mnt/sda/cbj/miniconda3/envs/Anomalydiffusion/bin/python
PREPARED="$REPO/results_for_gpt/plastic_bomo_chapter3_exp2_baseline_generation/prepared_v2_20261017_073039"
BASE_MODEL="$AD_SOURCE/models/ldm/text2img-large/model.ckpt"
BERT_SOURCE=/mnt/sda/cbj/GLIP-main/bert-base-uncased

finish() {
  code=$?
  echo "[已退出] AnomalyDiffusion 训练进程退出；退出码=$code"
}
trap finish EXIT

mkdir -p "$RUN_ROOT"
echo "[启动] AnomalyDiffusion V2 TRAIN；目标 optimizer steps=$MAX_STEPS；GPU=$GPU_ID"
echo "[运行中] 等待 GPU $GPU_ID 可用显存达到 ${MIN_FREE_MIB} MiB。"
while true; do
  free_mib="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
    break
  fi
  echo "[运行中] GPU $GPU_ID 当前空闲 ${free_mib:-unknown} MiB；继续等待。"
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

echo "[运行中] GPU 空闲条件通过；开始加载模型并执行训练。"
cd "$AD_SOURCE"
"$AD_PYTHON" -u main.py \
  --spatial_encoder_embedding \
  --data_enhance \
  --v2_manifest "$PREPARED/training_manifest.json" \
  --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml \
  --train true \
  --no-test true \
  --actual_resume "$BASE_MODEL" \
  --name Plastic_Bomo_exp2_ad \
  --logdir "$RUN_ROOT" \
  --gpus 0, \
  --seed 23 \
  --init_word anomaly \
  --scale_lr false \
  --max_steps "$MAX_STEPS" \
  data.params.batch_size=1 \
  data.params.num_workers=0 \
  lightning.modelcheckpoint.params.every_n_train_steps="$MAX_STEPS" \
  lightning.callbacks.image_logger.params.disabled=true
