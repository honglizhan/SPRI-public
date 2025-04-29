#!/bin/bash

set -o xtrace
cd ../../
echo "Current directory: $(pwd)"

# fill this in
MODELS=(
    "meta-llama/llama-3-1-70b-instruct"
    # "gpt-4o-mini"
    # "meta-llama/llama-3-8b-instruct"
    # "mistralai/mixtral-8x7b-instruct-v01"
    # "kaist-ai/prometheus-8x7b-v2"
)
BATCH_SIZE=512
enable_sampling_BigGenBench=false
export GENAI_KEY="<Input your GENAI key here>"
export OPENAI_API_KEY="<Input your OpenAI key here>"

for MODEL in "${MODELS[@]}"
do
    echo "Processing model: $MODEL"
    # ### ============================================================================== ###
    # # When generating intial rubrics, we use NO seed principles
    # ### ------ 01: Generate initial rubrics ------
    # python3 -m step2_gen_principles.01_gen_principles \
    #     -base_model_name=${MODEL} \
    #     -batch_size=${BATCH_SIZE} \
    #     -prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step2_01-rubric_gen.txt" \
    #     -seed_examples_path="" \
    #     -input_path="./[task]-BigGenBench/data/BiGGen-Bench-Results-[human_eval].jsonl" \
    #     -input_data_header="input" \
    #     -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[no_seed_examples]-[01_principles_raw].jsonl" \
    #     -task="BigGenBench" \
    #     -enable_sampling_BigGenBench=${enable_sampling_BigGenBench} \
    #     -reference_free

    # ### ------ 02: Critique and Refine the rubrics ------
    # python3 -m step2_gen_principles.02_refine_principles \
    #     -base_model_name=${MODEL} \
    #     -critic_model_name="kaist-ai/prometheus-8x7b-v2" \
    #     -batch_size=${BATCH_SIZE} \
    #     -critic_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step2_02-rubric_critique.txt" \
    #     -refine_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step2_02-rubric_refinement.txt" \
    #     -input_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[no_seed_examples]-[01_principles_raw].jsonl" \
    #     -input_data_header="input" \
    #     -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[no_seed_examples]-[02_principles_refined].jsonl" \
    #     -task="BigGenBench" \
    #     -reference_free

    # ### ------ 03: Generate Evaluation Results based on my generated rubrics ------
    # python3 -m step3_gen_responses.01_gen_init_responses \
    #     -base_model_name=${MODEL} \
    #     -batch_size=${BATCH_SIZE} \
    #     -input_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[no_seed_examples]-[02_principles_refined].jsonl" \
    #     -input_data_header="input" \
    #     -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[no_seed_examples]-[03_responses_raw].jsonl" \
    #     -task="BigGenBench" \
    #     -reference_free

    # ### ------ 04: Critique and Refine the Evaluation Results ------
    # python3 -m step3_gen_responses.02_prin_refine_responses \
    #     -base_model_name=${MODEL} \
    #     -critic_model_name="kaist-ai/prometheus-8x7b-v2" \
    #     -batch_size=${BATCH_SIZE} \
    #     -critic_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step3_02-eval_response_critique.txt" \
    #     -refine_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step3_02-eval_response_refinement.txt" \
    #     -input_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[no_seed_examples]-[03_responses_raw].jsonl" \
    #     -input_data_header="input" \
    #     -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[no_seed_examples]-[04_responses_refined].jsonl" \
    #     -task="BigGenBench" \
    #     -reference_free

    ### ============================================================================== ###
    # Default principles as seeds
    ### ------ 01: Generate initial rubrics ------
    python3 -m step2_gen_principles.01_gen_principles \
        -base_model_name=${MODEL} \
        -batch_size=${BATCH_SIZE} \
        -prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step2_01-rubric_gen.txt" \
        -seed_examples_path="./seeds/seed-principles-6.jsonl" \
        -input_path="./[task]-BigGenBench/data/BiGGen-Bench-Results-[human_eval].jsonl" \
        -input_data_header="input" \
        -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[6_default_principles]-[01_principles_raw].jsonl" \
        -task="BigGenBench" \
        -enable_sampling_BigGenBench=${enable_sampling_BigGenBench} \
        -reference_free

    ### ------ 02: Critique and Refine the rubrics ------
    python3 -m step2_gen_principles.02_refine_principles \
        -base_model_name=${MODEL} \
        -critic_model_name="kaist-ai/prometheus-8x7b-v2" \
        -batch_size=${BATCH_SIZE} \
        -critic_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step2_02-rubric_critique.txt" \
        -refine_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step2_02-rubric_refinement.txt" \
        -input_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[6_default_principles]-[01_principles_raw].jsonl" \
        -input_data_header="input" \
        -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[6_default_principles]-[02_principles_refined].jsonl" \
        -task="BigGenBench" \
        -reference_free

    ### ------ 03: Perform Eval based on my generated rubrics ------
    python3 -m step3_gen_responses.01_gen_init_responses \
        -base_model_name=${MODEL} \
        -batch_size=${BATCH_SIZE} \
        -input_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[6_default_principles]-[02_principles_refined].jsonl" \
        -input_data_header="input" \
        -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[6_default_principles]-[03_responses_raw].jsonl" \
        -task="BigGenBench" \
        -reference_free

    ### ------ 04: Critique and Refine the Evaluation Results ------
    python3 -m step3_gen_responses.02_prin_refine_responses \
        -base_model_name=${MODEL} \
        -critic_model_name="kaist-ai/prometheus-8x7b-v2" \
        -batch_size=${BATCH_SIZE} \
        -critic_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step3_02-eval_response_critique.txt" \
        -refine_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step3_02-eval_response_refinement.txt" \
        -input_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[6_default_principles]-[03_responses_raw].jsonl" \
        -input_data_header="input" \
        -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[6_default_principles]-[04_responses_refined].jsonl" \
        -task="BigGenBench" \
        -reference_free

    # ### ============================================================================== ###
    # # When generating intial rubrics, we use 3 gold rubrics from BigGenBench as seeds for each domain/i.e. capacity (i.e., seed-rubrics-[2024_11_28]-[reference_free]-[3_for_each_capacity].jsonl)
    # ### ------ 01: Generate initial rubrics ------
    # python3 -m step2_gen_principles.01_gen_principles \
    #     -base_model_name=${MODEL} \
    #     -batch_size=${BATCH_SIZE} \
    #     -prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step2_01-rubric_gen.txt" \
    #     -seed_examples_path="./seeds/seed-rubrics-[2024_11_28]-[reference_free]-[3_for_each_capacity].jsonl" \
    #     -input_path="./[task]-BigGenBench/data/BiGGen-Bench-Results-[human_eval].jsonl" \
    #     -input_data_header="input" \
    #     -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[3_for_each_capacity]-[01_principles_raw].jsonl" \
    #     -task="BigGenBench" \
    #     -enable_sampling_BigGenBench=${enable_sampling_BigGenBench} \
    #     -reference_free \
    #     -seed_examples_per_domain

    # ### ------ 02: Critique and Refine the rubrics ------
    # python3 -m step2_gen_principles.02_refine_principles \
    #     -base_model_name=${MODEL} \
    #     -critic_model_name="kaist-ai/prometheus-8x7b-v2" \
    #     -batch_size=${BATCH_SIZE} \
    #     -critic_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step2_02-rubric_critique.txt" \
    #     -refine_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step2_02-rubric_refinement.txt" \
    #     -input_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[3_for_each_capacity]-[01_principles_raw].jsonl" \
    #     -input_data_header="input" \
    #     -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[3_for_each_capacity]-[02_principles_refined].jsonl" \
    #     -task="BigGenBench" \
    #     -reference_free

    # ### ------ 03: Perform Eval based on my generated rubrics ------
    # python3 -m step3_gen_responses.01_gen_init_responses \
    #     -base_model_name=${MODEL} \
    #     -batch_size=${BATCH_SIZE} \
    #     -input_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[3_for_each_capacity]-[02_principles_refined].jsonl" \
    #     -input_data_header="input" \
    #     -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[3_for_each_capacity]-[03_responses_raw].jsonl" \
    #     -task="BigGenBench" \
    #     -reference_free

    # ### ------ 04: Critique and Refine the Evaluation Results ------
    # python3 -m step3_gen_responses.02_prin_refine_responses \
    #     -base_model_name=${MODEL} \
    #     -critic_model_name="kaist-ai/prometheus-8x7b-v2" \
    #     -batch_size=${BATCH_SIZE} \
    #     -critic_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step3_02-eval_response_critique.txt" \
    #     -refine_prompt_path="./instructions-[rubric]-v3-[reference_free]/instruct-step3_02-eval_response_refinement.txt" \
    #     -input_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[3_for_each_capacity]-[03_responses_raw].jsonl" \
    #     -input_data_header="input" \
    #     -output_path="./[task]-BigGenBench/outputs/${MODEL}/principle_instruct-[2024_11_28]-[reference_free]-[3_for_each_capacity]-[04_responses_refined].jsonl" \
    #     -task="BigGenBench" \
    #     -reference_free
done