import json
from tqdm import tqdm
from absl import app, flags
from utils import bam_llm_agents, tools
from termcolor import colored
import ast

from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE

from openai import OpenAI

FLAGS = flags.FLAGS
## ---- Model parameters ---
flags.DEFINE_string("base_model_name", "mistralai/mixtral-8x7b-instruct-v01", "")
flags.DEFINE_integer("batch_size", 24, "")
## ---- prompt path ---
flags.DEFINE_string("prompt_path", "", "")
flags.DEFINE_string("ABSOLLUTE_REFINE_PROMPT_path", "./[task]-BigGenBench/prompts/ABSOLLUTE_REFINE_PROMPT.txt", "")
flags.DEFINE_string("ABSOLUTE_PROMPT_path", "./[task]-BigGenBench/prompts/ABSOLUTE_PROMPT.txt", "")
flags.DEFINE_string("seed_examples_path", "", "")
## ---- input path ---
flags.DEFINE_string("input_path", "", "")
flags.DEFINE_string("filter_id_field", "id", "")
flags.DEFINE_string("input_data_header", "user_input", "")
## ---- output path ---
flags.DEFINE_string("output_path", "", "")
## ---- task type ---
flags.DEFINE_enum("task", "general-instruction-tuning", ["general-instruction-tuning", "reappraisal", "HumanEval-Fix", "BigGenBench"], "")

flags.DEFINE_boolean("reference_free", False, "")

def main(argv):
    if "gpt" in FLAGS.base_model_name:
        client = OpenAI()
    else:
        base_llm_agent = bam_llm_agents.bam_chat_agent(model_name=FLAGS.base_model_name)

    ## ---- load prompts & seed examples ---
    if FLAGS.task != "BigGenBench":
        prompt = open(FLAGS.prompt_path, "r").read()

    ## ---- load inputs ---
    input_lst = tools.read_jsonl(FLAGS.input_path)
    if FLAGS.task == "BigGenBench":
        input_lst = tools.filter_inputs_two_fields(input_lst, FLAGS.output_path, ["id", "model_name"])
    else:
        input_lst = tools.filter_inputs(input_lst, FLAGS.output_path, FLAGS.filter_id_field)
    print (colored(f"Total number of inputs: {len(input_lst)}", "green"))
    input_lst_chunks = tools.chunks(input_lst, FLAGS.batch_size)

    few_shot_examples = ""
    if FLAGS.seed_examples_path != "":
        seed_examples_raw_lst = tools.read_jsonl(FLAGS.seed_examples_path)
        ## ---- construct meta prompts for few shot examples ---
        for seed_example in seed_examples_raw_lst:
            few_shot_examples += f"### Question: {seed_example['question']}\n"
            few_shot_examples += f"### Principles: {seed_example['principle']}\n"
            few_shot_examples += f"### Response: {seed_example['bad_response']}\n\n"
    else: pass

    # print (colored("="*80, "red"))
    # print (colored("PROMPT:", "red"))
    # print (colored(f"""{prompt}\n\n{few_shot_examples}""", "green"))

    for chunk in tqdm(input_lst_chunks, total=(len(input_lst)//FLAGS.batch_size)+1):
        if "gpt" in FLAGS.base_model_name:
            batch_outputs = []

        formatted_batch_prompts = []
        for input in chunk:
            if FLAGS.task == "general-instruction-tuning":
                user_message = f"""### Question: {input[FLAGS.input_data_header]}\n### Principles: {input["principle_final"]}\n\n### Response:"""
            elif FLAGS.task == "reappraisal":
                user_message = f"""### Question: {input["Reddit Post"]}\n### Principles: {input["principle_final"]}\n\n### Response:"""
            elif FLAGS.task == "HumanEval-Fix":
                user_message = f"""### Question: You are a helpful assistant that debugs code. You will be provided with a buggy code snippet. Your task is to fix the code. Please do not generate any opening and closing remarks, nor explanations. Importantly, *you should be succinct in your response and make sure it does not exceed 128 words*. The fix code you provide should be ready to execute immediately, i.e. it should not be a mere continuation of the buggy code but rather a standalone code snippet. Wrap up the fixed code in your response in the following format: <fixed>[CODE]</fixed>. Buggy code: {input[FLAGS.input_data_header]}\n### Principles: {input["principle_final"]}\n\n### Response:"""
            elif FLAGS.task == "BigGenBench":
                if "llm_judge" in input["id"]:
                    GRADING_PROMPT = open(FLAGS.ABSOLLUTE_REFINE_PROMPT_path, "r").read()
                else:
                    GRADING_PROMPT = open(FLAGS.ABSOLUTE_PROMPT_path, "r").read()

                try:
                    my_rubric_dict = ast.literal_eval(input["principle_final"])
                    my_rubric = SCORE_RUBRIC_TEMPLATE.format(
                        criteria            = my_rubric_dict["criteria"],
                        score1_description  = my_rubric_dict["score1_description"],
                        score2_description  = my_rubric_dict["score2_description"],
                        score3_description  = my_rubric_dict["score3_description"],
                        score4_description  = my_rubric_dict["score4_description"],
                        score5_description  = my_rubric_dict["score5_description"]
                    )
                except:
                    my_rubric = input["principle_final"]

                if FLAGS.reference_free:
                    GRADING_PROMPT = GRADING_PROMPT.replace("###Reference Answer (Score 5):\n{reference_answer}\n\n", "").replace(""", a reference answer that gets a score of 5""", "")
                    user_message = GRADING_PROMPT.format(
                        instruction         = input[FLAGS.input_data_header],
                        response            = input["response"],
                        rubric              = my_rubric
                    )
                else:
                    user_message = GRADING_PROMPT.format(
                        instruction         = input[FLAGS.input_data_header],
                        response            = input["response"],
                        reference_answer    = input["reference_answer"],
                        rubric              = my_rubric
                    )

                input["formatted_BigGenBench_INIT_GRADING_PROMPT"] = user_message
            else:
                raise ValueError(f"Task {FLAGS.task} not recognized!")

            ### ---- appending formatted input to list w/ system prompts ----
            if "gpt" in FLAGS.base_model_name:
                if FLAGS.task == "BigGenBench":
                    my_message = [
                        {"role": "system", "content": '''You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance. The output of your assessment should strictly adhere to the following format: "<feedback>[YOUR FEEDBACK]</feedback><score>[YOUR SCORE]</score>"'''},
                        {"role": "user", "content": user_message}
                    ]
                else:
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
                if FLAGS.task == "BigGenBench":
                    formatted_batch_prompts.append(base_llm_agent.format_input(
                        system_prompt = '''You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance. The output of your assessment should strictly adhere to the following format: "<feedback>[YOUR FEEDBACK]</feedback><score>[YOUR SCORE]</score>"''',
                        user_prompt = user_message
                    ))
                else:
                    formatted_batch_prompts.append(base_llm_agent.format_input(
                        system_prompt = f"""{prompt}\n\n{few_shot_examples}""",
                        user_prompt = user_message
                    ))

        if "gpt" not in FLAGS.base_model_name:
            batch_outputs = base_llm_agent.batch_generate(formatted_batch_prompts)

        for idx, output in enumerate(batch_outputs):
            input = chunk[idx]
            # print (colored(input["user_input"], "yellow"))
            # print (colored(output, "red"))
            # print ()

            input["init_response"] = output.strip("\n#").strip()
            input["init_response_prompt"] = formatted_batch_prompts[idx]

            if FLAGS.task == "BigGenBench":
                input["init_response-[score]"] = tools.extract_biggenbench_eval_result(input["init_response"])

            with open(FLAGS.output_path, "a") as f:
                f.write(json.dumps(input) + "\n")

if __name__ == "__main__":
    app.run(main)