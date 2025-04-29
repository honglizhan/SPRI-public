import json
from tqdm import tqdm
from absl import app, flags
from utils import bam_llm_agents, tools
from termcolor import colored

FLAGS = flags.FLAGS
## ---- Model parameters ---
flags.DEFINE_string("base_model_name", "mistralai/mixtral-8x7b-instruct-v01", "")
flags.DEFINE_integer("batch_size", 24, "")
## ---- prompt path ---
flags.DEFINE_string("prompt_path", "", "")
# flags.DEFINE_string("seed_examples_path", "", "")
## ---- input path ---
flags.DEFINE_string("input_path", "", "")
## ---- output path ---
flags.DEFINE_string("output_path", "", "")


def main(argv):
    base_llm_agent = bam_llm_agents.bam_chat_agent(model_name=FLAGS.base_model_name)

    ## ---- load prompts & seed examples ---
    self_align_prompt = open(FLAGS.prompt_path, "r").read()

    ## ---- load inputs ---
    input_lst = tools.read_jsonl(FLAGS.input_path)
    input_lst = tools.filter_inputs(input_lst, FLAGS.output_path, "id")
    input_lst_chunks = tools.chunks(input_lst, FLAGS.batch_size)
    print ("# of examples to run:", len(input_lst))

    for chunk in tqdm(input_lst_chunks, total=(len(input_lst)//FLAGS.batch_size)+1):

        formatted_batch_prompts = []
        for input in chunk:

            user_message = f"User: {input['user_input']}"""

            if "mixtral" in FLAGS.base_model_name:
                formatted_batch_prompts.append(base_llm_agent.format_input_mixtral(
                    system_prompt = self_align_prompt,
                    user_prompt = user_message,
                ))
            elif "llama-3" in FLAGS.base_model_name:
                formatted_batch_prompts.append(base_llm_agent.format_input_llama3(
                    system_prompt = self_align_prompt,
                    user_prompt = user_message,
                ))
            else:
                raise ValueError(f"The current chat template does not support model {FLAGS.base_model_name}!")

        batch_outputs = base_llm_agent.batch_generate(formatted_batch_prompts)
        for idx, output in enumerate(batch_outputs):
            input = chunk[idx]
            # print (colored(input["user_input"], "yellow"))
            # print (colored(output, "red"))
            # print ()

            input["baseline-[self-align]_response"] = output.strip("\n#").strip()
            input["baseline-[self-align]_response_prompt"] = formatted_batch_prompts[idx]

            with open(FLAGS.output_path, "a") as f:
                f.write(json.dumps(input) + "\n")

if __name__ == "__main__":
    app.run(main)