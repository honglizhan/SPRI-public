#!/bin/bash

echo "Current directory: $(pwd)"
set -o xtrace
cd ../../

conda activate ibm
# fill this in
SOURCE_DATASET_NAME="unique_resort_eval_data.json"
OUTPUT_DATA_NAME="[resort_human_eval_30]"

export GENAI_KEY="<Input your GENAI key here>"
export OPENAI_API_KEY="<Input your OpenAI key here>"

BASE_MODEL_NAME="meta-llama/llama-3-1-70b-instruct"
SAVE_PATH_MODEL="llama-3-1-70b-instruct"

### ============================================================================== ###
# Step 2 01: Generate raw principles *given [few-shot examples & "question"]*
python3 -m step2_gen_principles.01_gen_principles \
    -base_model_name=${BASE_MODEL_NAME} \
    -batch_size=124 \
    -prompt_path="./instructions-[general-instruction-tuning]/instruct-step2_01-principle_gen.txt" \
    -seed_examples_path="./seeds/seed-principles-6.jsonl" \
    -input_path="./[task3]-reappraisal/RESORT_human_annotations/${SOURCE_DATASET_NAME}" \
    -filter_id_field="Reddit ID" \
    -output_path="./[task3]-reappraisal/outputs-[seed=Principle_Instruct]/${SAVE_PATH_MODEL}/${OUTPUT_DATA_NAME}-generated_principles_raw.jsonl" \
    -task="reappraisal"

# Step 2 02: Refine principles
python3 -m step2_gen_principles.02_refine_principles \
    -critic_model_name="kaist-ai/prometheus-8x7b-v2" \
    -base_model_name=${BASE_MODEL_NAME} \
    -batch_size=124 \
    -critic_prompt_path="./instructions-[general-instruction-tuning]/instruct-step2_02-principle_critique.txt" \
    -refine_prompt_path="./instructions-[general-instruction-tuning]/instruct-step2_02-principle_refinement.txt" \
    -input_path="./[task3]-reappraisal/outputs-[seed=Principle_Instruct]/${SAVE_PATH_MODEL}/${OUTPUT_DATA_NAME}-generated_principles_raw.jsonl" \
    -filter_id_field="Reddit ID" \
    -output_path="./[task3]-reappraisal/outputs-[seed=Principle_Instruct]/${SAVE_PATH_MODEL}/${OUTPUT_DATA_NAME}-generated_principles_refined.jsonl" \
    -task="reappraisal"


### ============================================================================== ###
# Step 3 01: Generate initial responses *given ["question" & "final_principles"]*
python3 -m step3_gen_responses.01_gen_init_responses \
    -base_model_name=${BASE_MODEL_NAME} \
    -batch_size=124 \
    -prompt_path="./instructions-[general-instruction-tuning]/instruct-step3_01-gen_init_responses.txt" \
    -seed_examples_path="" \
    -input_path="./[task3]-reappraisal/outputs-[seed=Principle_Instruct]/${SAVE_PATH_MODEL}/${OUTPUT_DATA_NAME}-generated_principles_refined.jsonl" \
    -filter_id_field="Reddit ID" \
    -output_path="./[task3]-reappraisal/outputs-[seed=Principle_Instruct]/${SAVE_PATH_MODEL}/${OUTPUT_DATA_NAME}-generated_init_responses.jsonl" \
    -task="reappraisal"

# Step 3 02: Generate refined responses *given ["question" & "final_principles" & "Critique"]*
python3 -m step3_gen_responses.02_prin_refine_responses \
    -critic_model_name="kaist-ai/prometheus-8x7b-v2" \
    -base_model_name=${BASE_MODEL_NAME} \
    -batch_size=124 \
    -critic_prompt_path="./instructions-[general-instruction-tuning]/instruct-step3_02-response_critique.txt" \
    -refine_prompt_path="./instructions-[general-instruction-tuning]/instruct-step3_02-response_refinement.txt" \
    -input_path="./[task3]-reappraisal/outputs-[seed=Principle_Instruct]/${SAVE_PATH_MODEL}/${OUTPUT_DATA_NAME}-generated_init_responses.jsonl" \
    -filter_id_field="Reddit ID" \
    -output_path="./[task3]-reappraisal/outputs-[seed=Principle_Instruct]/${SAVE_PATH_MODEL}/${OUTPUT_DATA_NAME}-generated_refined_responses.jsonl" \
    -task="reappraisal"