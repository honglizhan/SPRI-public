#!/bin/bash

echo "Current directory: $(pwd)"
set -o xtrace
cd ../../

# fill this in
SOURCE_DATASET_NAME="<your_dataset_name>"
OUTPUT_DATA_NAME="<your_output_data_name>"
export GENAI_KEY="<your_ibm_bam_key>"

### ============================================================================== ###
# Step 2 01: Generate raw principles *given [few-shot examples & "question"]*
python3 -m step2_gen_principles.01_gen_principles \
    -batch_size=512 \
    -base_model_name="mistralai/mixtral-8x7b-instruct-v01" \
    -prompt_path="./instructions/instruct-step2_01-principle_gen.txt" \
    -seed_examples_path="./seeds/seed-principles-6.jsonl" \
    -input_path="./source_dataset/${SOURCE_DATASET_NAME}.jsonl" \
    -output_path="./step2_gen_principles/outputs/${OUTPUT_DATA_NAME}-generated_principles_raw-mixtral-8x7b.jsonl"

# Step 2 02: Refine principles
python3 -m step2_gen_principles.02_refine_principles \
    -batch_size=512 \
    -critic_model_name="kaist-ai/prometheus-8x7b-v2" \
    -base_model_name="mistralai/mixtral-8x7b-instruct-v01" \
    -critic_prompt_path="./instructions/instruct-step2_02-principle_critique.txt" \
    -refine_prompt_path="./instructions/instruct-step2_02-principle_refinement.txt" \
    -input_path="./step2_gen_principles/outputs/${OUTPUT_DATA_NAME}-generated_principles_raw-mixtral-8x7b.jsonl" \
    -output_path="./step2_gen_principles/outputs/${OUTPUT_DATA_NAME}-generated_principles_refined-mixtral-8x7b.jsonl"



### ============================================================================== ###
# Step 3 01: Generate initial responses *given ["question" & "final_principles"]*
python3 -m step3_gen_responses.01_gen_init_responses \
    -batch_size=512 \
    -prompt_path="./instructions/instruct-step3_01-gen_init_responses.txt" \
    -seed_examples_path="" \
    -input_path="./step2_gen_principles/outputs/${OUTPUT_DATA_NAME}-generated_principles_refined-mixtral-8x7b.jsonl" \
    -output_path="./step3_gen_responses/outputs/${OUTPUT_DATA_NAME}-generated_init_responses-mixtral-8x7b.jsonl"

# Step 3 02: Generate refined responses *given ["question" & "final_principles" & "Critique"]*
python3 -m step3_gen_responses.02_prin_refine_responses \
    -batch_size=512 \
    -critic_model_name="kaist-ai/prometheus-8x7b-v2" \
    -base_model_name="mistralai/mixtral-8x7b-instruct-v01" \
    -critic_prompt_path="./instructions/instruct-step3_02-response_critique.txt" \
    -refine_prompt_path="./instructions/instruct-step3_02-response_refinement.txt" \
    -input_path="./step3_gen_responses/outputs/${OUTPUT_DATA_NAME}-generated_init_responses-mixtral-8x7b.jsonl" \
    -output_path="./step3_gen_responses/outputs/${OUTPUT_DATA_NAME}-generated_refined_responses-mixtral-8x7b.jsonl"



### ============================================================================== ###
# Step 4 01: Generate responses on the questions directly, w/o principles or few-shot examples
python3 -m step4_baseline_responses.01_gen_baseline_direct_responses \
    -batch_size=512 \
    -base_model_name="mistralai/mixtral-8x7b-instruct-v01" \
    -prompt_path="" \
    -seed_examples_path="" \
    -input_path="./step3_gen_responses/outputs/${OUTPUT_DATA_NAME}-generated_refined_responses-mixtral-8x7b.jsonl" \
    -output_path="./step4_baseline_responses/outputs/${OUTPUT_DATA_NAME}-generated_baseline_direct_responses-mixtral-8x7b.jsonl"