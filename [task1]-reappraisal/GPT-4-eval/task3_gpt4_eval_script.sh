#!/bin/bash

echo "Current directory: $(pwd)"
set -o xtrace
#cd ../

# conda activate ibm

export GENAI_KEY="<Input your GENAI key here>"
export OPENAI_API_KEY="<Input your OpenAI key here>"

MY_TASKS=(
    "1_standard_alignment"
    "2_empathy"
    "3_harmful"
    "4_factuality"
)

INDV_METHODS=(
    "vanilla"
    "+appr"
    "+cons"
    "+appr_+cons"
    "+cons_principle_instruct"
)

ITER_METHODS=(
    "self-refine"
    "+appr"
    "+cons"
    "+appr_+cons"
    "+cons_principle_instruct"
)

# fill this in
BASE_MODEL_NAME="gpt-4o-mini"

# # [seed=Principle_Instruct]
# for TASK in "${MY_TASKS[@]}"
# do
#     python gpt-4-reappraisal_eval.py \
#         -model="gpt-4-0613" \
#         -input_path="../outputs-[seed=Principle_Instruct]/${BASE_MODEL_NAME}/[resort_human_eval_30]-generated_refined_responses.jsonl" \
#         -output_path="./eval_outputs/${BASE_MODEL_NAME}/[seed=Principle_Instruct]/[resort_human_eval_30]-generated_refined_responses/" \
#         -eval_task=${TASK}
# done


# # [seed=RESORT_constitutions]
# for TASK in "${MY_TASKS[@]}"
# do
#     python gpt-4-reappraisal_eval.py \
#         -model="gpt-4-0613" \
#         -input_path="../outputs-[seed=RESORT_constitutions]/${BASE_MODEL_NAME}/[resort_human_eval_30]-generated_refined_responses.jsonl" \
#         -output_path="./eval_outputs/${BASE_MODEL_NAME}/[seed=RESORT_constitutions]/[resort_human_eval_30]-generated_refined_responses" \
#         -eval_task=${TASK}
# done

# # [seed=RESORT_constitutions]
# for TASK in "${MY_TASKS[@]}"
# do
#     python gpt-4-reappraisal_eval.py \
#         -model="gpt-4-0613" \
#         -input_path="../outputs-[seed=RESORT_constitutions]/${BASE_MODEL_NAME}/[resort_human_eval_30]-generated_refined_responses.jsonl" \
#         -output_path="./eval_outputs/${BASE_MODEL_NAME}/[seed=RESORT_constitutions]/[resort_human_eval_30]-generated_refined_responses" \
#         -eval_task=${TASK}
# done

# [seed=1_RESORT_constitutions_per_dimension]
for TASK in "${MY_TASKS[@]}"
do
    python gpt-4-reappraisal_eval.py \
        -model="gpt-4-0613" \
        -input_path="../outputs-[seed=1_RESORT_constitutions_per_dimension]/${BASE_MODEL_NAME}/[resort_human_eval_30]-generated_refined_responses.jsonl" \
        -output_path="./eval_outputs/${BASE_MODEL_NAME}/[seed=1_RESORT_constitutions_per_dimension]/[resort_human_eval_30]-generated_refined_responses" \
        -eval_task=${TASK}
done

# [seed=none]
for TASK in "${MY_TASKS[@]}"
do
    python gpt-4-reappraisal_eval.py \
        -model="gpt-4-0613" \
        -input_path="../outputs-[seed=none]/${BASE_MODEL_NAME}/[resort_human_eval_30]-generated_refined_responses.jsonl" \
        -output_path="./eval_outputs/${BASE_MODEL_NAME}/[seed=none]/[resort_human_eval_30]-generated_refined_responses" \
        -eval_task=${TASK}
done

# # [RESORT_baselines] = INDV
# for TASK in "${MY_TASKS[@]}"
# do
#     for METHOD in "${INDV_METHODS[@]}"
#     do
#         python gpt-4-reappraisal_eval.py \
#             -model="gpt-4-0613" \
#             -input_path="../outputs-[RESORT_baselines]/INDV/${METHOD}/unique_resort_eval_data/${BASE_MODEL_NAME}.jsonl" \
#             -output_path="./eval_outputs/${BASE_MODEL_NAME}/[RESORT_baselines]/INDV/${METHOD}" \
#             -eval_task=${TASK}
#     done
# done

# # [RESORT_baselines] = ITER
# for TASK in "${MY_TASKS[@]}"
# do
#     for METHOD in "${ITER_METHODS[@]}"
#     do
#         python gpt-4-reappraisal_eval.py \
#             -model="gpt-4-0613" \
#             -input_path="../outputs-[RESORT_baselines]/ITER/${METHOD}/unique_resort_eval_data/${BASE_MODEL_NAME}.jsonl" \
#             -output_path="./eval_outputs/${BASE_MODEL_NAME}/[RESORT_baselines]/ITER/${METHOD}" \
#             -eval_task=${TASK}
#     done
# done