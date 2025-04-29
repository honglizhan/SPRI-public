import os
import json
from tqdm import tqdm
from absl import app, flags
from termcolor import colored
from utils import bam_llm_agents, tools

from prometheus_eval.prompts import SCORE_RUBRIC_TEMPLATE

from openai import OpenAI

FLAGS = flags.FLAGS
flags.DEFINE_string("base_model_name", "meta-llama/llama-3-70b-instruct", "")
flags.DEFINE_integer("batch_size", 24, "")
flags.DEFINE_list("filter_id_two_fields", ["id", "model_name"], "")
flags.DEFINE_string("input_path", "", "")
flags.DEFINE_string("output_path", "", "")
flags.DEFINE_string("ABSOLUTE_PROMPT_path", "./[task]-BigGenBench/prompts/ABSOLUTE_PROMPT.txt", "")
flags.DEFINE_string("ABSOLLUTE_REFINE_PROMPT_path", "./[task]-BigGenBench/prompts/ABSOLLUTE_REFINE_PROMPT.txt", "")
flags.DEFINE_string("FLASK_RUBRICS_path", "./[task]-BigGenBench/prompts/FLASK_RUBRICS.jsonl", "")

flags.DEFINE_boolean("reference_free", False, "")


def load_flask_rubrics(capability, rubric_path):
    rubric_dict = {}
    with open(rubric_path, "r") as file:
        for line in file:
            entry = json.loads(line.strip())

            flask_task = entry["flask_task"]
            flask_rubric = entry["flask_rubric"]
            rubric_dict[flask_task] = flask_rubric

    if capability == "instruction_following":
        return rubric_dict.get("comprehension")
    elif capability == "grounding":
        return rubric_dict.get("factuality")
    elif capability == "reasoning":
        return rubric_dict.get("logical_robustness")
    elif capability == "planning":
        return rubric_dict.get("insightfulness")
    elif capability == "refinement":
        return rubric_dict.get("commonsense_understanding")
    elif capability == "safety":
        return rubric_dict.get("harmlessness")
    elif capability == "theory_of_mind":
        return rubric_dict.get("metacognition")
    elif capability == "tool_usage":
        return rubric_dict.get("comprehension")
    else:
        raise ValueError(f"capability: {capability} not found in rubric_dict")


def main(argv):

    if "gpt" in FLAGS.base_model_name or "o1" in FLAGS.base_model_name:
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

    # is_prometheus = False
    # if "prometheus" in FLAGS.base_model_name:
    #     is_prometheus = True

    for chunk in tqdm(input_lst_chunks, total=(len(input_lst)//FLAGS.batch_size)+1):
        if "gpt" in FLAGS.base_model_name:
            batch_outputs = []

        formatted_batch_prompts = []

        for input in chunk:
            if "llm_judge" in input["id"]:
                # if is_prometheus:
                #     continue
                GRADING_PROMPT = open(FLAGS.ABSOLLUTE_REFINE_PROMPT_path, "r").read()
            else:
                GRADING_PROMPT = open(FLAGS.ABSOLUTE_PROMPT_path, "r").read()

            if FLAGS.reference_free:
                GRADING_PROMPT = GRADING_PROMPT.replace("###Reference Answer (Score 5):\n{reference_answer}\n\n", "").replace(""", a reference answer that gets a score of 5""", "")
                user_message = GRADING_PROMPT.format(
                    instruction         = input["input"],
                    response            = input["response"],
                    rubric              = SCORE_RUBRIC_TEMPLATE.format(**load_flask_rubrics(input["capability"], FLAGS.FLASK_RUBRICS_path))
                )
            else:
                user_message = GRADING_PROMPT.format(
                    instruction         = input["input"],
                    response            = input["response"],
                    reference_answer    = input["reference_answer"],
                    rubric              = SCORE_RUBRIC_TEMPLATE.format(**load_flask_rubrics(input["capability"], FLAGS.FLASK_RUBRICS_path))
                )

            if "gpt" in FLAGS.base_model_name:
                my_message = [
                    {"role": "system", "content": '''You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance. The output of your assessment should strictly adhere to the following format: "<feedback>[YOUR FEEDBACK]</feedback><score>[YOUR SCORE]</score>"'''},
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
                    system_prompt = '''You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance. The output of your assessment should strictly adhere to the following format: "<feedback>[YOUR FEEDBACK]</feedback><score>[YOUR SCORE]</score>"''',
                    user_prompt = user_message,
                ))

        if "gpt" not in FLAGS.base_model_name:
            batch_outputs = base_llm_agent.batch_generate(formatted_batch_prompts)

        for idx, output in enumerate(batch_outputs):
            input = chunk[idx]
            input["[baseline]-flask"] = output.strip("\n#").strip()
            input["[baseline]-flask-score"] = tools.extract_biggenbench_eval_result(input["[baseline]-flask"])
            input["[baseline]-flask-prompt"] = formatted_batch_prompts[idx]

            with open(FLAGS.output_path, "a") as f:
                f.write(json.dumps(input) + "\n")

if __name__ == "__main__":
    app.run(main)