import os
import json
from tqdm import tqdm
from absl import app, flags
from termcolor import colored
from utils import bam_llm_agents, tools

from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE

from openai import OpenAI

FLAGS = flags.FLAGS
flags.DEFINE_string("base_model_name", "meta-llama/llama-3-1-70b-instruct", "")
flags.DEFINE_integer("batch_size", 24, "")
flags.DEFINE_list("filter_id_two_fields", ["id", "model_name"], "")
flags.DEFINE_string("input_path", "", "")
flags.DEFINE_string("output_path", "", "")
flags.DEFINE_string("MT_Bench_COARSE_PROMPT_SYSTEM_path", "./[task]-BigGenBench/prompts/MT-Bench-COARSE_PROMPT-[SYSTEM].txt", "")
flags.DEFINE_string("MT_Bench_COARSE_PROMPT_USER_path", "./[task]-BigGenBench/prompts/MT-Bench-COARSE_PROMPT-[USER].txt", "")

flags.DEFINE_boolean("reference_free", False, "")

def main(argv):

    if "gpt" in FLAGS.base_model_name:
        client = OpenAI()
    else:
        base_llm_agent = bam_llm_agents.bam_chat_agent(model_name=FLAGS.base_model_name)

    output_dir = os.path.dirname(FLAGS.output_path)
    os.makedirs(output_dir, exist_ok=True)

    ## ---- load inputs ---
    input_lst = tools.read_jsonl(FLAGS.input_path)
    input_lst = tools.filter_inputs_two_fields(input_lst, FLAGS.output_path, FLAGS.filter_id_two_fields)
    input_lst_chunks = tools.chunks(input_lst, FLAGS.batch_size)
    print (colored(f"# of examples to run: {len(input_lst)}", "green"))

    MT_Bench_COARSE_PROMPT_SYSTEM = open(FLAGS.MT_Bench_COARSE_PROMPT_SYSTEM_path, "r").read()
    MT_Bench_COARSE_PROMPT_USER = open(FLAGS.MT_Bench_COARSE_PROMPT_USER_path, "r").read()

    # is_prometheus = False
    # if "prometheus" in FLAGS.base_model_name:
    #     is_prometheus = True

    for chunk in tqdm(input_lst_chunks, total=(len(input_lst)//FLAGS.batch_size)+1):
        if "gpt" in FLAGS.base_model_name:
            batch_outputs = []

        formatted_batch_prompts = []

        for input in chunk:

            if FLAGS.reference_free:
                MT_Bench_COARSE_PROMPT_SYSTEM = MT_Bench_COARSE_PROMPT_SYSTEM.replace("You will be given a reference answer and the assistant’s answer.", "You will be given the assistant’s answer.")
                MT_Bench_COARSE_PROMPT_USER = MT_Bench_COARSE_PROMPT_USER.replace("### Reference Answer:\n{orig_reference_answer}\n", "")
                user_message = MT_Bench_COARSE_PROMPT_USER.format(
                    orig_instruction         = input["input"],
                    orig_response            = input["response"]
                )
            else:
                user_message = MT_Bench_COARSE_PROMPT_USER.format(
                    orig_instruction         = input["input"],
                    orig_reference_answer    = input["reference_answer"],
                    orig_response            = input["response"]
                )

            if "gpt" in FLAGS.base_model_name:
                my_message = [
                    {"role": "system", "content": MT_Bench_COARSE_PROMPT_SYSTEM},
                    {"role": "user", "content": user_message}
                ]

                completion = client.chat.completions.create(
                    model=FLAGS.base_model_name.replace("gpt-o1", "o1"),
                    messages=my_message
                )
                batch_outputs.append(completion.choices[0].message.content)
                formatted_batch_prompts.append(my_message)
            else:
                formatted_batch_prompts.append(base_llm_agent.format_input(
                    system_prompt = MT_Bench_COARSE_PROMPT_SYSTEM,
                    user_prompt = user_message,
                ))

        if "gpt" not in FLAGS.base_model_name:
            batch_outputs = base_llm_agent.batch_generate(formatted_batch_prompts)

        for idx, output in enumerate(batch_outputs):
            input = chunk[idx]
            input["[baseline]-vanilla-MT_Bench"] = output.strip("\n#").strip()
            input["[baseline]-vanilla-MT_Bench-score"] = tools.extract_biggenbench_eval_result(input["[baseline]-vanilla-MT_Bench"])
            input["[baseline]-vanilla-MT_Bench-prompt"] = formatted_batch_prompts[idx]

            with open(FLAGS.output_path, "a") as f:
                f.write(json.dumps(input) + "\n")

if __name__ == "__main__":
    app.run(main)