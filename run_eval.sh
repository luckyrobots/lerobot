#!/bin/bash

# Directory where your models are stored
MODEL_DIR="outputs/train"
# List of model subdirectories (edit as needed)
MODELS=(
    "act/089000/pretrained_model"
)

# Environment type (edit as needed)
ENV_TYPE="luckyworld"
# Output directory for evaluation results
OUTPUT_DIR="outputs/eval"
# Number of episodes to evaluate
N_EPISODES=50
# Batch size for evaluation
BATCH_SIZE=1
# Device to use (cpu or cuda)
DEVICE="cuda"

# Loop over each model and run evaluation
for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"
    OUT_PATH="${OUTPUT_DIR}/${MODEL_NAME}"
    mkdir -p "$OUT_PATH"
    echo "Evaluating model: $MODEL_PATH"
    python lerobot/scripts/eval.py \
        --policy.path="$MODEL_PATH" \
        --env.type="$ENV_TYPE" \
        --eval.batch_size="$BATCH_SIZE" \
        --eval.use_async_envs=false \
        --eval.n_episodes="$N_EPISODES" \
        --policy.device="$DEVICE" \
        --output_dir="$OUT_PATH"
    echo "Results saved to $OUT_PATH"
    echo "----------------------------------------"
done
