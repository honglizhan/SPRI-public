#!/bin/bash

cd ../RESORT_pipeline
echo "Current directory: $(pwd)"
set -o xtrace

export GENAI_KEY="<Input your GENAI key here>"
export OPENAI_API_KEY="<Input your OpenAI key here>"

BASE_MODEL_NAME="meta-llama/llama-3-1-70b-instruct"

INDV_BASELINE_METHODS=(
    "vanilla"
    "+appr"
    "+cons"
    "+appr_+cons"
)

ITER_BASELINE_METHODS=(
    "self-refine"
    "+appr"
    "+cons"
    "+appr_+cons"
)

for METHOD in "${INDV_BASELINE_METHODS[@]}"
do
python3 -m [bam]-pipeline-INDV \
    -base_model_name=${BASE_MODEL_NAME} \
    -input_data_path="../RESORT_human_annotations/unique_resort_eval_data.json" \
    -output_path="../outputs-[RESORT_baselines]/INDV" \
    -experiment_mode=${METHOD}
done

for METHOD in "${ITER_BASELINE_METHODS[@]}"
do
python3 -m [bam]-pipeline-ITER \
    -base_model_name=${BASE_MODEL_NAME} \
    -input_data_path="../RESORT_human_annotations/unique_resort_eval_data.json" \
    -output_path="../outputs-[RESORT_baselines]/ITER" \
    -experiment_mode=${METHOD}
done

### ------ Run INDV +cons with principle-instruct principles ------
python3 -m [bam]-pipeline-INDV \
    -base_model_name=${BASE_MODEL_NAME} \
    -input_data_path="../RESORT_human_annotations/unique_resort_eval_data.json" \
    -output_path="../outputs-[RESORT_baselines]/INDV" \
    -path_to_reappraisal_guidance="./prompts/reappraisal_guidance-[principle_instruct_seeds]/reappraisal_guidance.txt" \
    -experiment_mode="+cons_principle_instruct"

### ------ Run ITER +cons with principle-instruct principles ------
python3 -m [bam]-pipeline-ITER \
    -base_model_name=${BASE_MODEL_NAME} \
    -input_data_path="../RESORT_human_annotations/unique_resort_eval_data.json" \
    -output_path="../outputs-[RESORT_baselines]/ITER" \
    -path_to_reappraisal_guidance="./prompts/reappraisal_guidance-[principle_instruct_seeds]/reappraisal_guidance.txt" \
    -experiment_mode="+cons_principle_instruct"