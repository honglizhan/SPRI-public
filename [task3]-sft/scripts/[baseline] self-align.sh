#!/bin/bash

cd ../
echo "Current directory: $(pwd)"
set -o xtrace

# fill this in
DATASET_NAME="llama3-mix_instruct-val_2k-sub_columns"
export GENAI_KEY="<your_ibm_bam_key>"

### ============================================================================== ###
python3 -m step4_baseline_responses.03_gen_[self-align]_responses \
    -batch_size=1024 \
    -base_model_name="meta-llama/llama-3-70b-instruct" \
    -prompt_path="./step4_baseline_responses/watson_self_align_prompt.txt" \
    -input_path="./[final-lora-finetuning] my_datasets/${DATASET_NAME}.jsonl" \
    -output_path="./step4_baseline_responses/outputs_[03-self-align]/[self-align]_${DATASET_NAME}.jsonl"