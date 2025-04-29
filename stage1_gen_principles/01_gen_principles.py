import json
from tqdm import tqdm
from absl import app, flags
from utils import bam_llm_agents, tools
from termcolor import colored
import os
import random
from openai import OpenAI


FLAGS = flags.FLAGS
## ---- Model parameters ---
flags.DEFINE_string("base_model_name", "mistralai/mixtral-8x7b-instruct-v01", "")
flags.DEFINE_integer("batch_size", 24, "")
## ---- prompt & seed examples path ---
flags.DEFINE_string("prompt_path", "", "")
flags.DEFINE_string("seed_examples_path", "", "")
## ---- input path ---
flags.DEFINE_string("input_path", "", "")
flags.DEFINE_string("filter_id_field", "id", "")
flags.DEFINE_string("input_data_header", "user_input", "")
## ---- output path ---
flags.DEFINE_string("output_path", "", "")
## ---- task type ---
flags.DEFINE_enum("task", "sft", ["sft", "reappraisal", "BigGenBench"], "")

### ---- BigGenBench specific flags ---
flags.DEFINE_boolean("reference_free", False, "")
flags.DEFINE_boolean("seed_examples_per_domain", False, "")


def main(argv):
    random.seed(42)
    output_dir = os.path.dirname(FLAGS.output_path)
    os.makedirs(output_dir, exist_ok=True)

    if "gpt" in FLAGS.base_model_name:
        client = OpenAI()
    else:
        base_llm_agent = bam_llm_agents.bam_chat_agent(model_name=FLAGS.base_model_name)

    ## ---- load prompts ---
    prompt = open(FLAGS.prompt_path, "r").read()

    ## ---- load inputs ---
    if FLAGS.input_path.split(".")[-1] == "jsonl":
        input_lst = tools.read_jsonl(FLAGS.input_path)
    elif FLAGS.input_path.split(".")[-1] == "json":
        input_lst = tools.read_json(FLAGS.input_path)
    else:
        raise ValueError(f"Input file type {FLAGS.input_path.split('.')[-1]} not recognized!")

    if FLAGS.task == "BigGenBench":
        input_lst = tools.filter_inputs_two_fields(input_lst, FLAGS.output_path, ["id", "model_name"])
    else:
        input_lst = tools.filter_inputs(input_lst, FLAGS.output_path, FLAGS.filter_id_field)
    print (colored(f"Total number of inputs: {len(input_lst)}", "green"))
    input_lst_chunks = tools.chunks(input_lst, FLAGS.batch_size)

    # print (colored("="*80, "red"))
    # print (colored("PROMPT:", "red"))
    # print (colored(f"""{prompt}\n\n{few_shot_examples}""", "green"))

    for chunk in tqdm(input_lst_chunks, total=(len(input_lst)//FLAGS.batch_size)+1):
        if "gpt" in FLAGS.base_model_name:
            batch_outputs = []

        formatted_batch_prompts = []
        for input in chunk:

            few_shot_examples = ""
            if FLAGS.seed_examples_path == "":
                prompt = prompt.replace(" When phrasing the scoring rubric, follow these examples:", "").replace(" When phrasing principles, follow these examples:", "")
            else:
                ### ------ Loading seed examples ------ ###
                seed_examples_raw_lst = tools.read_jsonl(FLAGS.seed_examples_path)

                ## ---- construct meta prompts for few shot examples ---
                # Using 6 Principle-Instruct principles
                if "seed-principles" in FLAGS.seed_examples_path:
                    for seed_example in seed_examples_raw_lst:
                        few_shot_examples += f"### Question: {seed_example['question']}\n"
                        few_shot_examples += f"### Principles: {seed_example['principle']}\n\n"
                # Using 6 RESORT constitutions
                elif "seed-reappraisal_constitutions" in FLAGS.seed_examples_path:
                    if FLAGS.seed_examples_per_domain:
                        seed_examples_raw_lst = [item for item in seed_examples_raw_lst if str(item["appraisal_dimension_id"]) == str(input["appraisal_dimension_id"])]
                    for seed_example in seed_examples_raw_lst:
                        few_shot_examples += f"### Question: {seed_example['reddit_post']}\n"
                        few_shot_examples += f"### Response Goal: {seed_example['appraisal_dimension_aim']}\n"
                        few_shot_examples += f"### Principles: {seed_example['constitution']}\n\n"
                # Using gold BigGenBench rubrics
                elif "seed-rubrics" in FLAGS.seed_examples_path:
                    if FLAGS.seed_examples_per_domain:
                        seed_examples_raw_lst = [item for item in seed_examples_raw_lst if item["capability"] == input["capability"]]
                    for seed_example in seed_examples_raw_lst:
                        few_shot_examples += f"### Input: {seed_example['system_prompt']}\n\n{seed_example['input']}\n"
                        if FLAGS.reference_free == False:
                            few_shot_examples += f"### Reference Answer: {seed_example['reference_answer']}\n"
                        few_shot_examples += f"### Rubric: {str(seed_example['rubric'])}\n\n"
                else:
                    raise ValueError(f"Seed examples path {FLAGS.seed_examples_path} not recognized!")


            if FLAGS.task == "sft":
                user_message = f"""### Question: {input[FLAGS.input_data_header]}\n### Principles:"""
            elif FLAGS.task == "reappraisal":
                user_message = f"""### Question: {input["Reddit Post"]}\n### Response Goal: {input['appraisal_dimension_aim']}\n### Principles:"""
            elif FLAGS.task == "HumanEval-Fix":
                user_message = f"""You are a helpful assistant that debugs code. You will be provided with a buggy code snippet. Your task is to fix the code. Please do not generate any opening and closing remarks, nor explanations. Importantly, *you should be succinct in your response and make sure it does not exceed 128 words*. Buggy code: {input[FLAGS.input_data_header]}\n### Principles:"""
            elif FLAGS.task == "BigGenBench":
                if FLAGS.reference_free:
                    user_message = f"""### Input: {input['system_prompt']}\n\n{input["input"]}\n\n### Rubric:"""
                else:
                    user_message = f"""### Input: {input['system_prompt']}\n\n{input["input"]}\n\n### Reference Answer: {input["reference_answer"]}\n\n### Rubric:"""
            else:
                raise ValueError(f"Task {FLAGS.task} not recognized!")

            if "gpt" in FLAGS.base_model_name:
                my_message = [
                    {"role": "system", "content": f"""{prompt}\n\n{few_shot_examples}"""},
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
                    system_prompt = f"""{prompt}\n\n{few_shot_examples}""",
                    user_prompt = user_message
                ))

        if "gpt" not in FLAGS.base_model_name:
            batch_outputs = base_llm_agent.batch_generate(formatted_batch_prompts)

        for idx, output in enumerate(batch_outputs):
            input = chunk[idx]
            input["principle_raw"] = output.strip("\n#").strip() # un-critiqued & unrefined principle
            input["principle_raw_prompt"] = formatted_batch_prompts[idx]

            with open(FLAGS.output_path, "a") as f:
                f.write(json.dumps(input) + "\n")

if __name__ == "__main__":
    app.run(main)