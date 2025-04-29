#==============#
# Importations #
#==============#
import os
import json
from absl import app, flags
from tqdm import tqdm
from pipeline_components import prompt_loading
import RESORT_utils, RESORT_bam_llm_agents
from openai import OpenAI


#=============#
# Define Args #
#=============#
FLAGS = flags.FLAGS

### ------ Define the LLM ------
flags.DEFINE_string("base_model_name", "meta-llama/Llama-2-13b-chat-hf", "Specify the LLM.")

### ------ Loading input/output data path ------
flags.DEFINE_string("input_data_path", "./source_data/r_anger.jsonl", "")
flags.DEFINE_string("output_path", "./model_outputs/individual_guided_reappraisal", "")

### ------ Loading prompt path ------
flags.DEFINE_string("path_to_appraisal_questions", "./prompts/appraisal_questions.txt", "File with all appraisal questions.")
flags.DEFINE_string("path_to_reappraisal_guidance", "./prompts/reappraisal_guidance.txt", "File with all re-appraisal guidance.")

### ------ Define experiment mode ------
flags.DEFINE_enum("experiment_mode", "vanilla", ["vanilla", "+appr", "+cons", "+cons_principle_instruct", "+appr_+cons"], "Specify experiment mode.")
flags.DEFINE_list("dimensions", [1, 4, 6, 7, 8, 23], "Specify the dimensions to look at.")

flags.DEFINE_boolean("use_bam", True, "")



def main(argv):

    if "gpt" in FLAGS.base_model_name:
        ChatAgent = RESORT_bam_llm_agents.openai_chat_agent(model_name=FLAGS.base_model_name, client=OpenAI())
    elif FLAGS.use_bam == True:
        ChatAgent = RESORT_bam_llm_agents.bam_chat_agent(model_name=FLAGS.base_model_name)
    else:
        raise ValueError("Model name not supported!")

    my_system_prompt = """Respond with a response in the format requested by the user. Do not acknowledge my request with "sure" or in any other way besides going straight to the answer."""
    model_name = FLAGS.base_model_name.split('/')[-1]

    dimensions_df = RESORT_utils.get_prompts(FLAGS.path_to_appraisal_questions, FLAGS.path_to_reappraisal_guidance)

    path = f"""{FLAGS.output_path}/{FLAGS.experiment_mode}/{FLAGS.input_data_path.split('/')[-1].split('.')[0]}"""
    if not os.path.exists(path):
        os.makedirs(path)
    output_jsonl_file_path = f"{path}/{model_name}.jsonl"

    #==================#
    # Elicit responses #
    #==================#
    eval_set = RESORT_utils.read_json(FLAGS.input_data_path)
    eval_set = RESORT_utils.filter_inputs(eval_set, output_jsonl_file_path, input_fields=["Reddit ID", "appraisal_dimension_id"])

    for row in tqdm(eval_set):
        if FLAGS.experiment_mode == "vanilla":
            baseline_vanilla_prompt = "Please help the narrator of the text reappraise the situation. Your response should be concise and brief."
            step2_input, step2_output = ChatAgent.chat(
                system_message = my_system_prompt,
                user_message = f"""[Text] {row['Reddit Post']}\n\n[Question] {baseline_vanilla_prompt}""")

            ChatAgent.reset()

            row["reappraisal_prompt"] = str(step2_input)
            row["reappraisal_output"] = str(step2_output)

        elif FLAGS.experiment_mode == "+cons":
            dim = row["appraisal_dimension_id"]
            # for dim in FLAGS.dimensions:
            dim_prompt = f"""Please help the narrator of the text reappraise the situation. {dimensions_df.reappraisal_guidance[dim]} Your response should be concise and brief."""
            step2_input, step2_output = ChatAgent.chat(
                system_message = my_system_prompt,
                user_message = f"""[Text] {row['Reddit Post']}\n\n[Question] {dim_prompt}""")
            ChatAgent.reset()

            row[f"reappraisal_prompt_dim_{dim}"] = str(step2_input)
            row[f"reappraisal_output_dim_{dim}"] = str(step2_output)

        elif FLAGS.experiment_mode == "+cons_principle_instruct":
            dim = row["appraisal_dimension_id"]
            # for dim in FLAGS.dimensions:
            dim_prompt = f"""Please help the narrator of the text reappraise the situation. {dimensions_df.reappraisal_guidance[dim]} Your response should be concise and brief."""
            step2_input, step2_output = ChatAgent.chat(
                system_message = my_system_prompt,
                user_message = f"""[Text] {row['Reddit Post']}\n\n[Question] {dim_prompt}""")
            ChatAgent.reset()

            row[f"reappraisal_prompt_dim_{dim}"] = str(step2_input)
            row[f"reappraisal_output_dim_{dim}"] = str(step2_output)

        elif FLAGS.experiment_mode == "+appr":
            baseline_prompt = "Based on the analysis above, please help the narrator of the text reappraise the situation. Your response should be concise and brief."
            dim = row["appraisal_dimension_id"]
            # for dim in FLAGS.dimensions:
            prompt_step1 = prompt_loading.build_appraisal_prompt(
                text = row['Reddit Post'],
                appraisal_q = dimensions_df.appraisal_questions[dim],
            )

            # Step 1: Elicit appraisals
            step1_input, step1_output = ChatAgent.chat(
                system_message = my_system_prompt,
                user_message = prompt_step1)
            row[f"appraisal_prompt_dim_{dim}"] = str(step1_input)
            row[f"appraisal_output_dim_{dim}"] = str(step1_output)

            # Step 2: Ask baseline reappraisal prompt
            step2_input, step2_output = ChatAgent.chat(
                system_message = None,
                user_message = baseline_prompt)
            ChatAgent.reset()

            row[f"reappraisal_prompt_dim_{dim}"] = str(step2_input)
            row[f"reappraisal_output_dim_{dim}"] = str(step2_output)

        elif FLAGS.experiment_mode == "+appr_+cons":
            dim = row["appraisal_dimension_id"]
            # for dim in FLAGS.dimensions:
            prompt_step1 = prompt_loading.build_appraisal_prompt(
                text = row['Reddit Post'],
                appraisal_q = dimensions_df.appraisal_questions[dim],
            )

            # Step 1: Elicit appraisals
            step1_input, step1_output = ChatAgent.chat(
                system_message = my_system_prompt,
                user_message = prompt_step1)
            row[f"appraisal_prompt_dim_{dim}"] = str(step1_input)
            row[f"appraisal_output_dim_{dim}"] = str(step1_output)

            # Step 2: Ask to reappraise
            dim_prompt = f"""Based on the analysis above, please help the narrator of the text reappraise the situation. {dimensions_df.reappraisal_guidance[dim]} Your response should be concise and brief."""
            step2_input, step2_output = ChatAgent.chat(
                system_message = None,
                user_message = dim_prompt)
            ChatAgent.reset()

            row[f"reappraisal_prompt_dim_{dim}"] = str(step2_input)
            row[f"reappraisal_output_dim_{dim}"] = str(step2_output)

        else: raise ValueError

        with open(output_jsonl_file_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        ChatAgent.reset()

if __name__ == "__main__":
    app.run(main)