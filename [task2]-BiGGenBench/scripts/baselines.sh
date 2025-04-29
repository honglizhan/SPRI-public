#!/bin/bash

set -o xtrace
cd ../../
echo "Current directory: $(pwd)"

# fill this in
MODELS=(
    # "meta-llama/llama-3-1-70b-instruct"
    # "meta-llama/llama-3-8b-instruct"
    # "mistralai/mixtral-8x7b-instruct-v01"
    "kaist-ai/prometheus-8x7b-v2"
    # "gpt-4o-mini"
)
BATCH_SIZE=512
export GENAI_KEY="<Input your GENAI key here>"
export OPENAI_API_KEY="<Input your OpenAI key here>"

for epoch in {1..10}
do
    echo "Starting epoch: $epoch"
    for MODEL in "${MODELS[@]}"
    do
        echo "Processing model: $MODEL"

        cd "[task]-BigGenBench"
        python calc_pearson_corr.py -model_name ${MODEL}  -input_file_name [baseline]-[reference_free]-0-[gold_rubrics]  # -drop_null_scores
        # python calc_pearson_corr.py -model_name ${MODEL}  -input_file_name [baseline]-[reference_free]-1-[vanilla]   -drop_null_scores
        # python calc_pearson_corr.py -model_name ${MODEL}  -input_file_name [baseline]-[reference_free]-2-[self_refine]   -drop_null_scores
        python calc_pearson_corr.py -model_name ${MODEL}  -input_file_name [baseline]-[reference_free]-3-[MT_Bench] #  -drop_null_scores
        # python calc_pearson_corr.py -model_name ${MODEL}  -input_file_name [baseline]-[reference_free]-4-[flask]   -drop_null_scores
        python calc_pearson_corr.py -model_name ${MODEL}  -input_file_name [baseline]-[reference_free]-5-[seed_principles_as_rubrics] #  -drop_null_scores
        cd ../


        ### ============================================================================== ###
        ### ***0-direct_eval***
        python3 -m [task]-BigGenBench.0-direct_eval \
            -base_model_name=${MODEL} \
            -batch_size=${BATCH_SIZE} \
            -input_path="./[task]-BigGenBench/data/BiGGen-Bench-Results-[human_eval].jsonl" \
            -output_path="./[task]-BigGenBench/outputs/${MODEL}/[baseline]-[reference_free]-0-[gold_rubrics].jsonl" \
            -reference_free

        ### ***1-[baseline]-vanilla***
        python3 -m [task]-BigGenBench.1-[baseline]-vanilla \
            -base_model_name=${MODEL} \
            -batch_size=${BATCH_SIZE} \
            -input_path="./[task]-BigGenBench/data/BiGGen-Bench-Results-[human_eval].jsonl" \
            -output_path="./[task]-BigGenBench/outputs/${MODEL}/[baseline]-[reference_free]-1-[vanilla].jsonl" \
            -reference_free

        ### ***2-[baseline]-self_refine***
        python3 -m [task]-BigGenBench.2-[baseline]-self_refine \
            -base_model_name=${MODEL} \
            -batch_size=${BATCH_SIZE} \
            -input_path="./[task]-BigGenBench/data/BiGGen-Bench-Results-[human_eval].jsonl" \
            -output_path="./[task]-BigGenBench/outputs/${MODEL}/[baseline]-[reference_free]-2-[self_refine].jsonl" \
            -reference_free


        ### ============================================================================== ###
        ### ***3-[baseline]-MT_Bench***
        python3 -m [task]-BigGenBench.3-[baseline]-MT_Bench \
            -base_model_name=${MODEL} \
            -batch_size=${BATCH_SIZE} \
            -input_path="./[task]-BigGenBench/data/BiGGen-Bench-Results-[human_eval].jsonl" \
            -output_path="./[task]-BigGenBench/outputs/${MODEL}/[baseline]-[reference_free]-3-[MT_Bench].jsonl" \
            -reference_free

        ### ============================================================================== ###
        ### ***4-[baseline]-flask***
        python3 -m [task]-BigGenBench.4-[baseline]-flask \
            -base_model_name=${MODEL} \
            -batch_size=${BATCH_SIZE} \
            -input_path="./[task]-BigGenBench/data/BiGGen-Bench-Results-[human_eval].jsonl" \
            -output_path="./[task]-BigGenBench/outputs/${MODEL}/[baseline]-[reference_free]-4-[flask].jsonl" \
            -reference_free

        ### ============================================================================== ###
        ### ***5-[baseline]-seed_principles_as_rubrics***
        python3 -m [task]-BigGenBench.5-[baseline]-seed_principles_as_rubrics \
            -base_model_name=${MODEL} \
            -batch_size=${BATCH_SIZE} \
            -input_path="./[task]-BigGenBench/data/BiGGen-Bench-Results-[human_eval].jsonl" \
            -output_path="./[task]-BigGenBench/outputs/${MODEL}/[baseline]-[reference_free]-5-[seed_principles_as_rubrics].jsonl" \
            -reference_free
    done
done