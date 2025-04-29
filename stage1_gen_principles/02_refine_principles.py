import json
from tqdm import tqdm
from absl import app, flags
from utils import bam_llm_agents, tools
from termcolor import colored

from openai import OpenAI

FLAGS = flags.FLAGS
## ---- Model parameters ---
flags.DEFINE_string("base_model_name", "mistralai/mixtral-8x7b-instruct-v01", "")
flags.DEFINE_string("critic_model_name", "kaist-ai/prometheus-8x7b-v2", "")
flags.DEFINE_integer("batch_size", 24, "")
## ---- prompts path ---
flags.DEFINE_string("critic_prompt_path", "", "")
flags.DEFINE_string("refine_prompt_path", "", "")
## ---- input path ---
flags.DEFINE_string("input_path", "", "")
flags.DEFINE_string("filter_id_field", "id", "")
flags.DEFINE_string("input_data_header", "user_input", "")
## ---- output path ---
flags.DEFINE_string("output_path", "", "")
## ---- task type ---
flags.DEFINE_enum("task", "general-instruction-tuning", ["general-instruction-tuning", "reappraisal", "HumanEval-Fix", "BigGenBench"], "")

flags.DEFINE_boolean("reference_free", False, "")
flags.DEFINE_integer("min_eval_score", 4, "")

def main(argv):
    if "gpt" in FLAGS.base_model_name:
        client = OpenAI()
    else:
        base_llm_agent = bam_llm_agents.bam_chat_agent(model_name=FLAGS.base_model_name)
    critic_llm_agent = bam_llm_agents.bam_chat_agent(model_name=FLAGS.critic_model_name)

    ## ---- load instruction prompts ---
    critic_inst_prompt = open(FLAGS.critic_prompt_path, "r").read()
    refine_inst_prompt = open(FLAGS.refine_prompt_path, "r").read()

    ## ---- load inputs ---
    input_lst = tools.read_jsonl(FLAGS.input_path)

    if FLAGS.task == "BigGenBench":
        input_lst = tools.filter_inputs_two_fields(input_lst, FLAGS.output_path, ["id", "model_name"])
    else:
        input_lst = tools.filter_inputs(input_lst, FLAGS.output_path, FLAGS.filter_id_field)
    print (colored(f"Total number of inputs: {len(input_lst)}", "green"))
    input_lst_chunks = tools.chunks(input_lst, FLAGS.batch_size)

    # print (colored("="*80, "red"))
    # print (colored("Critic PROMPT:", "red"))
    # print (colored(critic_inst_prompt, "green"))
    # print (colored("="*80, "red"))
    # print (colored("Refinement PROMPT:", "red"))
    # print (colored(refine_inst_prompt, "green"))

    for chunk in tqdm(input_lst_chunks, total=(len(input_lst)//FLAGS.batch_size)+1):
        remaining_inputs = chunk.copy()
        for input in chunk:
            input["iteration_principle"] = 0
            input["principle_final"] = input["principle_raw"]

        # Continue refining until all inputs are either accepted or reach max iterations
        while remaining_inputs:
            batch_critic_prompts = []
            for input in remaining_inputs:
                iteration = input["iteration_principle"] + 1
                input["iteration_principle"] = iteration

                if FLAGS.task == "general-instruction-tuning":
                    replacements = {
                        'orig_question': input[FLAGS.input_data_header],
                        "orig_principle": input["principle_final"],
                    }
                elif FLAGS.task == "reappraisal":
                    replacements = {
                        'orig_question': input["Reddit Post"],
                        'response_goal': input["appraisal_dimension_aim"],
                        "orig_principle": input["principle_final"],
                    }
                elif FLAGS.task == "HumanEval-Fix":
                    replacements = {
                        'orig_question': f"""You are a helpful assistant that debugs code. You will be provided with a buggy code snippet. Your task is to fix the code. Please do not generate any opening and closing remarks, nor explanations. Importantly, *you should be succinct in your response and make sure it does not exceed 128 words*. Buggy code: {input[FLAGS.input_data_header]}""",
                        "orig_principle": input["principle_final"],
                    }
                elif FLAGS.task == "BigGenBench":
                    if FLAGS.reference_free:
                        replacements = {
                            'orig_input': f"""{input['system_prompt']}\n\n{input["input"]}""",
                            "orig_rubric": input["principle_final"],
                        }
                    else:
                        replacements = {
                            'orig_input': f"""{input['system_prompt']}\n\n{input["input"]}""",
                            "reference_answer": input["reference_answer"],
                            "orig_rubric": input["principle_final"],
                        }
                else:
                    raise ValueError(f"Task {FLAGS.task} not recognized!")

                # Prepare batch prompts for critique
                critic_prompt = critic_inst_prompt.format(**replacements)
                batch_critic_prompts.append(critic_prompt)

            # Generate critiques in batch
            batch_critic_prompts = [
                critic_llm_agent.format_input_prometheus(
                    system_prompt = None,
                    user_prompt = p) for p in batch_critic_prompts
            ]
            critiques = critic_llm_agent.batch_generate(batch_critic_prompts)

            next_remaining_inputs = []
            refinement_prompts = []

            for idx, input in enumerate(remaining_inputs):
                critique = critiques[idx].strip("\n#").strip()
                input[f"principles_critique_iter_{input['iteration_principle']}"] = critique
                input[f"principles_critique_iter_{input['iteration_principle']}_prompt"] = batch_critic_prompts[idx]
                eval_score = tools.extract_evaluation_score(critique)
                eval_feedback = tools.extract_evaluation_feedback(critique)

                try:
                    if eval_score >= FLAGS.min_eval_score:
                        input["principles_critique_final"] = critique
                    else:
                        if input["iteration_principle"] < 4: # setting max_iterations to 4
                            # Prepare batch prompts for refinement
                            if FLAGS.task == "general-instruction-tuning":
                                user_message = f"""### Question: {input[FLAGS.input_data_header]}\n### Principles: {input["principle_final"]}\n### Feedback: {eval_feedback}\n### Refined Principles:"""
                            elif FLAGS.task == "reappraisal":
                                user_message = f"""### Question: {input["Reddit Post"]}\n### Principles: {input["principle_final"]}\n### Feedback: {eval_feedback}\n### Refined Principles:"""
                            elif FLAGS.task == "HumanEval-Fix":
                                user_message = f"""You are a helpful assistant that debugs code. You will be provided with a buggy code snippet. Your task is to fix the code. Please do not generate any opening and closing remarks, nor explanations. Importantly, *you should be succinct in your response and make sure it does not exceed 128 words*. Buggy code: {input[FLAGS.input_data_header]}\n### Principles: {input["principle_final"]}\n### Feedback: {eval_feedback}\n### Refined Principles:"""
                            elif FLAGS.task == "BigGenBench":
                                if FLAGS.reference_free:
                                    user_message = f"""### Input: {input['system_prompt']}\n\n{input["input"]}\n\n### Rubric: {input["principle_final"]}\n\n### Feedback: {eval_feedback}\n\n### Refined Rubric:"""
                                else:
                                    user_message = f"""### Input: {input['system_prompt']}\n\n{input["input"]}\n\n### Reference Answer: {input["reference_answer"]}\n\n### Rubric: {input["principle_final"]}\n\n### Feedback: {eval_feedback}\n\n### Refined Rubric:"""
                            else:
                                raise ValueError(f"Task {FLAGS.task} not recognized!")

                            refinement_prompts.append(user_message)
                            next_remaining_inputs.append(input)
                        else:
                            input["principles_critique_final"] = critique
                except Exception as e:
                    print (colored(e, "blue"))
                    input[f"error_in_principle_critique_iter_{input['iteration_principle']}"] = str(e)
                    #break

            if refinement_prompts:
                if "gpt" in FLAGS.base_model_name:
                    refined_principles = []
                    for p in refinement_prompts:
                        my_message = [
                            {"role": "system", "content": refine_inst_prompt},
                            {"role": "user", "content": p}
                        ]
                        completion = client.chat.completions.create(
                            model=FLAGS.base_model_name.replace("gpt-o1", "o1"),
                            messages=my_message
                        )
                        refined_principles.append(completion.choices[0].message.content.strip("\n#").strip())

                else:
                    # Generate refined principles in batch
                    refinement_prompts = [
                        base_llm_agent.format_input(
                            system_prompt = refine_inst_prompt,
                            user_prompt = p) for p in refinement_prompts
                    ]

                    refined_principles = base_llm_agent.batch_generate(refinement_prompts)

                for idx, input in enumerate(next_remaining_inputs):
                    input[f"principle_refined_iter_{input['iteration_principle']}"] = refined_principles[idx].strip("\n#").strip()
                    input[f"principle_refined_iter_{input['iteration_principle']}_prompt"] = refinement_prompts[idx]
                    input["principle_final"] = refined_principles[idx]

            remaining_inputs = next_remaining_inputs

        # Save results to the output file
        for input in chunk:
            with open(FLAGS.output_path, "a") as f:
                f.write(json.dumps(input) + "\n")

if __name__ == "__main__":
    app.run(main)