#!/bin/bash

cd ../../
echo "Current directory: $(pwd)"
set -o xtrace

# fill this in
DATASET_NAMES=(
    "llama3-dolly-train_10k"
    "llama3-dolly-val_2k"
    "llama3-mix_instruct-train_50k-sub_columns"
    "llama3-mix_instruct-val_2k-sub_columns"
)
export GENAI_KEY="<your_ibm_bam_key>"

for DATASET_NAME in "${DATASET_NAMES[@]}"
do
    ### ============================================================================== ###
    python3 -m step4_baseline_responses.05_gen_[seed_principles]_responses \
        -base_model_name="meta-llama/llama-3-70b-instruct" \
        -batch_size=512 \
        -input_path="./source_dataset/[final-lora-finetuning] my_datasets/${DATASET_NAME}.jsonl" \
        -output_path="./step4_baseline_responses/outputs_[6-seed-principles]/${DATASET_NAME}.jsonl"
done