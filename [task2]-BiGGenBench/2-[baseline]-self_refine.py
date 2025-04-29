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

    # is_prometheus = False
    # if "prometheus" in FLAGS.base_model_name:
    #     is_prometheus = True

    for chunk in tqdm(input_lst_chunks, total=(len(input_lst)//FLAGS.batch_size)+1):
        if "gpt" not in FLAGS.base_model_name:
            init_formatted_batch_prompts = []


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
                    rubric              = '''How well does the response address the instruction? Please rate on a scale of 1 to 5, where 1 stands for "not at all" and 5 stands for "perfectly". The output of your assessment should strictly adhere to the following format: "<feedback>[YOUR FEEDBACK]</feedback><score>[YOUR SCORE]</score>"'''
                )
            else:
                user_message = GRADING_PROMPT.format(
                    instruction         = input["input"],
                    response            = input["response"],
                    reference_answer    = input["reference_answer"],
                    rubric              = '''How well does the response address the instruction? Please rate on a scale of 1 to 5, where 1 stands for "not at all" and 5 stands for "perfectly". The output of your assessment should strictly adhere to the following format: "<feedback>[YOUR FEEDBACK]</feedback><score>[YOUR SCORE]</score>"'''
                )

            if "gpt" in FLAGS.base_model_name:
                init_message = [
                    {"role": "system", "content": '''You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance. The output of your assessment should strictly adhere to the following format: "<feedback>[YOUR FEEDBACK]</feedback><score>[YOUR SCORE]</score>"'''},
                    {"role": "user", "content": user_message}
                ]

                completion = client.chat.completions.create(
                    model=FLAGS.base_model_name.replace("gpt-o1", "o1"),
                    messages=init_message
                )

                input["[self-refine]-iteration-0"] = completion.choices[0].message.content.strip()
                input["[self-refine]-iteration-0-score"] = tools.extract_biggenbench_eval_result(input["[self-refine]-iteration-0"])
                input["[self-refine]-iteration-0-prompt"] = init_message

                for refine_step in range(5):
                    refinement_message = [
                        {"role": "system", "content": '''You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance. Please refine the feedback and score to improve clarity, precision, and alignment with the rubric. The output of your assessment should strictly adhere to the following format: "<feedback>[YOUR FEEDBACK]</feedback><score>[YOUR SCORE]</score>"'''},
                        {"role": "user", "content": user_message + "\n\n\n" + input[f"[self-refine]-iteration-{refine_step}"] + '''\n\n\nPlease refine the feedback and score to improve clarity, precision, and alignment with the rubric. The output of your assessment should strictly adhere to the following format: "<feedback>[YOUR FEEDBACK]</feedback><score>[YOUR SCORE]</score>"'''}
                    ]
                    refinement_completion = client.chat.completions.create(
                        model=FLAGS.base_model_name.replace("gpt-o1", "o1"),
                        messages=refinement_message
                    )
                    input[f"[self-refine]-iteration-{refine_step+1}"] = refinement_completion.choices[0].message.content.strip()
                    input[f"[self-refine]-iteration-{refine_step+1}-score"] = tools.extract_biggenbench_eval_result(input[f"[self-refine]-iteration-{refine_step+1}"])
                    input[f"[self-refine]-iteration-{refine_step+1}-prompt"] = refinement_message

            else:
                init_message = base_llm_agent.format_input(
                    system_prompt='''You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance. The output of your assessment should strictly adhere to the following format: "<feedback>[YOUR FEEDBACK]</feedback><score>[YOUR SCORE]</score>"''',
                    user_prompt=user_message,
                )
                init_formatted_batch_prompts.append(init_message)


        if "gpt" not in FLAGS.base_model_name:
            init_batch_outputs = base_llm_agent.batch_generate(init_formatted_batch_prompts)

            for idx, output in enumerate(init_batch_outputs):
                input = chunk[idx]
                input["[self-refine]-iteration-0"] = output.strip("\n#").strip()
                input["[self-refine]-iteration-0-score"] = tools.extract_biggenbench_eval_result(input["[self-refine]-iteration-0"])
                input["[self-refine]-iteration-0-prompt"] = init_formatted_batch_prompts[idx]

            current_outputs = [output.strip("\n#").strip() for output in init_batch_outputs]

            for refine_step in range(5):
                refinement_prompts = []
                for output in current_outputs:
                    refinement_prompts.append(base_llm_agent.format_input(
                        system_prompt='''You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance. Please refine the feedback and score to improve clarity, precision, and alignment with the rubric. The output of your assessment should strictly adhere to the following format: "<feedback>[YOUR FEEDBACK]</feedback><score>[YOUR SCORE]</score>"''',
                        user_prompt=output,
                    ))
                current_outputs = base_llm_agent.batch_generate(refinement_prompts)
                current_outputs = [output.strip("\n#").strip() for output in current_outputs]

                for idx, output in enumerate(current_outputs):
                    input = chunk[idx]
                    input[f"[self-refine]-iteration-{refine_step+1}"] = output.strip("\n#").strip()
                    input[f"[self-refine]-iteration-{refine_step+1}-score"] = tools.extract_biggenbench_eval_result(input[f"[self-refine]-iteration-{refine_step+1}"])
                    input[f"[self-refine]-iteration-{refine_step+1}-prompt"] = init_formatted_batch_prompts[idx]

        for input in chunk:
            with open(FLAGS.output_path, "a") as f:
                f.write(json.dumps(input) + "\n")

if __name__ == "__main__":
    app.run(main)