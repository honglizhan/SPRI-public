####################
### Importations ###
####################
import os
import json
import pandas as pd
from tqdm import tqdm
from absl import app, flags
from termcolor import colored
from openai import OpenAI, AzureOpenAI
from chat_agents import OpenAIAgent
import eval_prompts


#=============#
# Define Args #
#=============#
FLAGS = flags.FLAGS

flags.DEFINE_string("model", "gpt-4-0613", "Specify the LLM.")
flags.DEFINE_integer("seed", 36, "Setting seed for reproducible outputs.")

## ---- Hyper-parameters for LLMs ---
flags.DEFINE_float("temperature", 0.1, "The value used to modulate the next token probabilities.")
flags.DEFINE_integer("max_tokens", 256, "Setting max tokens to generate.")

## ---- input path ---
flags.DEFINE_string("input_path", "../outputs/[llama3]-[resort_human_eval_30]-generated_refined_responses.jsonl", "")

## ---- output path ---
flags.DEFINE_string("output_path", "./eval_outputs/[llama3]-[resort_human_eval_30]-generated_refined_responses", "")

## ---- For GPT models, choose to use Azure or openai keys ---
flags.DEFINE_boolean("use_azure", False, "Use Azure OpenAI API instead of OpenAI API.")

## ---- choose eval task ---
flags.DEFINE_enum(
    "eval_task",
    default="1_standard_alignment",
    enum_values=["1_standard_alignment", "2_empathy", "3_harmful", "4_factuality"],
    help="Evaluation task selection."
)


def read_jsonl(jsonl_path):
    """ returns a list of dictionaries """
    with open(jsonl_path) as f:
        lst = f.readlines()
        lst = [
            json.loads(line.strip()) for line in lst if line.strip()
        ]
    return lst


def filter_inputs(input_lst, output_path, input_fields: list = ["Reddit ID", "appraisal_dimension_id"]):
    assert len(input_fields) == 2, "input_fields must be a list of two fields."
    filtered_input_lst = input_lst[:]

    ## ---- filter out already-generated questions ---
    if os.path.isfile(output_path):
        print (colored(f"Output file found at {output_path}. Filtering out already generated questions.", "red"))
        output_file = read_jsonl(output_path)
        output_tuples = [(dct[input_fields[0]], dct[input_fields[1]]) for dct in output_file]
        for input in input_lst:
            input_tuple = (input[input_fields[0]], input[input_fields[1]])
            if input_tuple in output_tuples:
                filtered_input_lst.remove(input)
        print (colored(f"Filtered out {len(input_lst) - len(filtered_input_lst)} questions.", "green"))
    else:
        print (colored(f"Output file not found at {output_path}.", "red"))
    return filtered_input_lst



def main(argv):

    #################################
    ### Establishing Class Agents ###
    #################################

    if "gpt" in FLAGS.model:

        if FLAGS.use_azure == True:
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2023-05-15"
            )
        else:
            client = OpenAI()

        ChatAgent = OpenAIAgent(
            FLAGS,
            client = client,
            new_system_prompt = """Respond with a response in the format requested by the user. Do not acknowledge my request with "sure" or in any other way besides going straight to the answer.""",
        )

    else: raise ValueError

    #==================#
    # Elicit responses #
    #==================#

    # read jsonl format input
    if FLAGS.input_path.endswith(".jsonl"):
        eval_set = read_jsonl(FLAGS.input_path)
    else:
        raise ValueError("Input file must be in jsonl format.")

    if not os.path.exists(FLAGS.output_path):
        os.makedirs(FLAGS.output_path)

    jsonl_file_output_path = f"{FLAGS.output_path}/eval-{FLAGS.eval_task}.jsonl"
    # print (f"Output file path: {jsonl_file_output_path}")
    eval_set = filter_inputs(eval_set, jsonl_file_output_path, ["Reddit ID", "appraisal_dimension_id"])

    for eval_input in tqdm(eval_set):

        post = eval_input["Reddit Post"]
        #post_id = eval_input["Reddit ID"]

        appraisal_dimension_id = eval_input["appraisal_dimension_id"]
        reappraisal_aim = eval_input["appraisal_dimension_aim"]
        reappraisal_standard = eval_input["standards"]
        reappraisal_response = eval_input["response_final"]

        ## ---- Criterion 1: standard alignment ---
        if FLAGS.eval_task == "1_standard_alignment":
            eval_prompt = eval_prompts.build_eval_prompt_1_standard_alignment(
                post = post,
                reappraisal_aim = reappraisal_aim,
                reappraisal_standard = reappraisal_standard,
                reappraisal = reappraisal_response,
            )

        ## ---- Criterion 2: empathy ---
        elif FLAGS.eval_task == "2_empathy":
            eval_prompt = eval_prompts.build_eval_prompt_2_empathy(
                post = post,
                reappraisal = reappraisal_response,
            )

        ## ---- Criterion 3: harmful ---
        elif FLAGS.eval_task == "3_harmful":
            eval_prompt = eval_prompts.build_eval_prompt_3_harmful(
                post = post,
                reappraisal = reappraisal_response
            )

        ## ---- Criterion 3: harmful ---
        elif FLAGS.eval_task == "3_harmful":
            eval_prompt = eval_prompts.build_eval_prompt_3_harmful(
                post = post,
                reappraisal = reappraisal_response
            )

        ## ---- Criterion 4: factuality ---
        elif FLAGS.eval_task == "4_factuality":
            eval_prompt = eval_prompts.build_eval_prompt_4_factuality(
                post = post,
                reappraisal = reappraisal_response
            )

        else:
            raise ValueError(f"Invalid eval task: {FLAGS.eval_task}")

        raw_eval = ChatAgent.chat(eval_prompt)
        eval_input[f"evaluation_raw_output-[dim_{appraisal_dimension_id}]-[{FLAGS.eval_task}]"] = raw_eval
        print (colored(raw_eval, "yellow"))
        ChatAgent.reset()


        with open(jsonl_file_output_path, "a") as f:
            f.write(json.dumps(eval_input) + "\n")

if __name__ == "__main__":
    app.run(main)