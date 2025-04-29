import json
from tqdm import tqdm
from absl import app, flags
from utils import bam_llm_agents, tools
from termcolor import colored
import os

FLAGS = flags.FLAGS
## ---- Model parameters ---
flags.DEFINE_string("base_model_name", "mistralai/mixtral-8x7b-instruct-v01", "")
flags.DEFINE_integer("batch_size", 24, "")
## ---- prompt path ---
flags.DEFINE_string("prompt_path", "", "")
flags.DEFINE_string("seed_examples_path", "", "")
## ---- input path ---
flags.DEFINE_string("input_path", "", "")
flags.DEFINE_string("filter_id_field", "id", "")
## ---- output path ---
flags.DEFINE_string("output_path", "", "")
flags.DEFINE_enum("task", "HumanEval-Fix", ["HumanEval-Fix", "general"], "")


def main(argv):
    base_llm_agent = bam_llm_agents.bam_chat_agent(model_name=FLAGS.base_model_name)

    output_dir = os.path.dirname(FLAGS.output_path)
    os.makedirs(output_dir, exist_ok=True)

    ## ---- load prompts & seed examples ---
    if FLAGS.prompt_path != "":
        prompt = open(FLAGS.prompt_path, "r").read()

    ## ---- load inputs ---
    input_lst = tools.read_jsonl(FLAGS.input_path)
    input_lst = tools.filter_inputs(input_lst, FLAGS.output_path, FLAGS.filter_id_field)
    input_lst_chunks = tools.chunks(input_lst, FLAGS.batch_size)
    print ("# of examples to run:", len(input_lst))

    print (colored("="*80, "red"))
    print (colored("PROMPT:", "red"))

    if FLAGS.task == "general":
        system_message = """You are a helpful assistant. Respond to the question directly. Please do not generate any opening and closing remarks, nor explanations. Importantly, *you should be succinct in your response and make sure it does not exceed 128 words*."""
        user_message = "### Question: {INPUT}\n\n### Response:"
    elif FLAGS.task == "HumanEval-Fix":
        system_message = """You are a helpful assistant that debugs code. You will be provided with a buggy code snippet. Your task is to fix the code. Please do not generate any opening and closing remarks, nor explanations. Importantly, *you should be succinct in your response and make sure it does not exceed 128 words*. The fix code you provide should be ready to execute immediately, i.e. it should not be a mere continuation of the buggy code but rather a standalone code snippet. Wrap up the fixed code in your response in the following format: <fixed>[CODE]</fixed>."""
        user_message = "### Buggy Code: {INPUT}\n\n### Fixed Code:"
    else:
        raise ValueError(f"Task {FLAGS.task} is not supported!")

    for chunk in tqdm(input_lst_chunks, total=(len(input_lst)//FLAGS.batch_size)+1):

        formatted_batch_prompts = []
        for input in chunk:

            formatted_batch_prompts.append(base_llm_agent.format_input(
                system_prompt = system_message,
                user_prompt = user_message.format(INPUT=input["user_input"]),
            ))

        batch_outputs = base_llm_agent.batch_generate(formatted_batch_prompts)
        for idx, output in enumerate(batch_outputs):
            input = chunk[idx]
            print (colored(input["user_input"], "yellow"))
            print (colored(output, "red"))
            print ()

            input["baseline-direct_response"] = output.strip("\n#").strip()
            input["baseline-direct_response_prompt"] = formatted_batch_prompts[idx]

            with open(FLAGS.output_path, "a") as f:
                f.write(json.dumps(input) + "\n")

if __name__ == "__main__":
    app.run(main)