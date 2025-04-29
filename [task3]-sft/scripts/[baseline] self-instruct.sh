#!/bin/bash

echo "Current directory: $(pwd)"
set -o xtrace
cd ../

#conda activate ibm
# fill this in
OUTPUT_DATA_NAME="llama3-dolly-train_10k"
export GENAI_KEY="<your_ibm_bam_key>"

### ============================================================================== ###
# Step 4 02: [Baseline] Generate self-instruct responses on the training dataset we have already
python3 -m step4_baseline_responses.02_gen_self-instruct \
    -batch_size=128 \
    -base_model_name="meta-llama/llama-3-70b-instruct" \
    -input_path="./[final-lora-finetuning] my_datasets/${OUTPUT_DATA_NAME}.jsonl" \
    -output_path="./step4_baseline_responses/outputs_[02-self-instruct]/[self-instruct]_${OUTPUT_DATA_NAME}.jsonl"