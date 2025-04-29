import os
import json
from tqdm import tqdm
from absl import app, flags
from utils import bam_llm_agents, tools


FLAGS = flags.FLAGS
## ---- Model parameters ---
flags.DEFINE_string("base_model_name", "mistralai/mixtral-8x7b-instruct-v01", "")
flags.DEFINE_integer("batch_size", 24, "")
## ---- prompt path ---
# flags.DEFINE_string("prompt_path", "", "")
# flags.DEFINE_string("seed_examples_path", "", "")
## ---- input path ---
flags.DEFINE_string("input_path", "", "")
## ---- output path ---
flags.DEFINE_string("output_path", "", "")

flags.DEFINE_integer("max_iterations", 1, "Number of iterations to run the self-refine algorithm.")

import re

def extract_evaluation_score(text):
    """ Return the evaluation score from the critic model on the intermediate responses. """
    pattern = r'\[?RESULT\]?:\s*([1-5])'
    match = re.search(pattern, text)

    if match:
        return int(match.group(1))
    else:
        return None


def main(argv):
    base_llm_agent = bam_llm_agents.bam_chat_agent(model_name=FLAGS.base_model_name)

    ## ---- load inputs ---
    input_lst = tools.read_jsonl(FLAGS.input_path)
    input_lst = tools.filter_inputs(input_lst, FLAGS.output_path, "id")
    input_lst_chunks = tools.chunks(input_lst, FLAGS.batch_size)
    print ("# of examples to run:", len(input_lst))

    output_dir = os.path.dirname(FLAGS.output_path)
    os.makedirs(output_dir, exist_ok=True)

    def stop_condition(feedback, iteration, max_iterations=5):
        feedback_score = extract_evaluation_score(feedback)
        # print (feedback)
        # Define the stop condition here (e.g., based on feedback score or max iterations)
        # if "sufficient" in feedback or "Sufficient" in feedback or iteration >= max_iterations:
        try:
            if feedback_score >= 4: # sometimes there are parsing issues
                return True
        except:
            pass

        if iteration > max_iterations:
            return True
        return False


    for chunk in tqdm(input_lst_chunks, total=(len(input_lst)//FLAGS.batch_size)+1):

        ### ------ Step 1: Initial response generation ------
        formatted_batch_prompts_turn_0 = []
        for input in chunk:
            # self-refine *Initial Output* prompt format
            initial_output_prompt = f"### Question: {input['user_input']}\n\n### Response:"""

            if "mixtral" in FLAGS.base_model_name:
                formatted_batch_prompts_turn_0.append(base_llm_agent.format_input_mixtral(
                    system_prompt = "You are a helpful assistant. Respond to the question directly. Please do not generate any opening and closing remarks, nor explanations. Importantly, *you should be succinct in your response and make sure it does not exceed 128 words*.",
                    user_prompt = initial_output_prompt,
                ))
            elif "llama-3" in FLAGS.base_model_name:
                formatted_batch_prompts_turn_0.append(base_llm_agent.format_input_llama3(
                    system_prompt = "You are a helpful assistant. Respond to the question directly. Please do not generate any opening and closing remarks, nor explanations. Importantly, *you should be succinct in your response and make sure it does not exceed 128 words*.",
                    user_prompt = initial_output_prompt,
                ))
            else:
                raise ValueError(f"The current chat template does not support model {FLAGS.base_model_name}!")

        # Generate batch initial responses (y0)
        batch_outputs_turn_0 = base_llm_agent.batch_generate(formatted_batch_prompts_turn_0)
        for idx, output in enumerate(batch_outputs_turn_0):
            input = chunk[idx]

            input["self-refine_response-[turn_0]"] = output.strip("\n#").strip()
            input["self-refine_response-[turn_0]_prompt"] = formatted_batch_prompts_turn_0[idx]
            input["self-refine_response-[final]"] = input["self-refine_response-[turn_0]"]

        # Track stop condition flags for each example
        stop_flags = [False] * len(chunk)

        ### ------ Step 2: Feedback-Refinement loop ------
        for t in range(1, FLAGS.max_iterations+1):
            # Step 2.1: Generate feedback for the entire batch
            feedback_batch_prompts = []
            for idx, input in enumerate(chunk):
                if not stop_flags[idx]:  # Only generate feedback for examples that haven't met the stopping criteria
                    y_t = input.get(f"self-refine_response-[turn_{t}]", input["self-refine_response-[turn_0]"])
                    feedback_prompt = f"### Question: {input['user_input']}\n\n### Response: {y_t}\n\n### Feedback:"
                    if "mixtral" in FLAGS.base_model_name:
                        feedback_batch_prompts.append(base_llm_agent.format_input_mixtral(
                            # system_prompt="You will be provided with an instruction and its corresponding response. Assess how well the response fulfills the instruction based on accuracy, completeness, and relevance. If the response fully addresses the instruction without any issues, reply with 'sufficient'. Otherwise, provide clear and constructive feedback for improvement.",
                            system_prompt="You will be provided with an instruction and its corresponding response. On a scale of 1 to 5, assess how well the response fulfills the instruction based on accuracy, completeness, and relevance. Provide clear and constructive feedback for improvement. The output format should look as follows: \'Feedback: (write a feedback based on the evaluation criteria) [RESULT]: (write an integer number between 1 and 5)\'",
                            user_prompt=feedback_prompt,
                        ))
                    elif "llama-3" in FLAGS.base_model_name:
                        feedback_batch_prompts.append(base_llm_agent.format_input_llama3(
                            # system_prompt="You will be provided with an instruction and its corresponding response. Assess how well the response fulfills the instruction based on accuracy, completeness, and relevance. If the response fully addresses the instruction without any issues, reply with 'sufficient'. Otherwise, provide clear and constructive feedback for improvement.",
                            system_prompt="You will be provided with an instruction and its corresponding response. On a scale of 1 to 5, assess how well the response fulfills the instruction based on accuracy, completeness, and relevance. Provide clear and constructive feedback for improvement. The output format should look as follows: \'Feedback: (write a feedback based on the evaluation criteria) [RESULT]: (write an integer number between 1 and 5)\'",
                            user_prompt=feedback_prompt,
                        ))
                    else:
                        raise ValueError(f"The current chat template does not support model {FLAGS.base_model_name}!")

            # Generate batch feedback only for active examples
            feedback_batch_outputs = base_llm_agent.batch_generate(feedback_batch_prompts)

            feedback_idx = 0
            for idx, input in enumerate(chunk):
                if not stop_flags[idx]:  # Only update feedback for examples that haven't stopped
                    feedback = feedback_batch_outputs[feedback_idx]
                    input[f"feedback-[turn_{t}]"] = feedback.strip("\n#").strip()
                    input[f"feedback-[turn_{t}]_prompt"] = feedback_batch_prompts[feedback_idx]
                    feedback_idx += 1

                    # Check stop condition for each example
                    stop_flags[idx] = stop_condition(input[f"feedback-[turn_{t}]"], t, max_iterations=FLAGS.max_iterations)

            # Step 2.2: If all examples meet the stop condition, break the loop
            if all(stop_flags):
                break

            # Step 2.3: Refine the entire batch based on feedback, but only for examples that haven't stopped
            refined_batch_prompts = []
            for idx, input in enumerate(chunk):
                if not stop_flags[idx]:  # Only refine responses that haven't stopped
                    y_t = input.get(f"self-refine_response-[turn_{t-1}]", input["self-refine_response-[turn_0]"])
                    feedback = input[f"feedback-[turn_{t}]"]
                    refine_prompt = f"### Question: {input['user_input']}\n\n### Original Response: {y_t}\n\n### Feedback: {feedback}\n\n### Refined Response:"
                
                    if "mixtral" in FLAGS.base_model_name:
                        refined_batch_prompts.append(base_llm_agent.format_input_mixtral(
                            system_prompt="Refine the original response based on the feedback provided.",
                            user_prompt=refine_prompt,
                        ))
                    elif "llama-3" in FLAGS.base_model_name:
                        refined_batch_prompts.append(base_llm_agent.format_input_llama3(
                            system_prompt="Refine the original response based on the feedback provided.",
                            user_prompt=refine_prompt,
                        ))
                    else:
                        raise ValueError(f"The current chat template does not support model {FLAGS.base_model_name}!")

            # Generate refined responses for the batch (only for active examples)
            refined_batch_outputs = base_llm_agent.batch_generate(refined_batch_prompts)

            refine_idx = 0
            for idx, input in enumerate(chunk):
                if not stop_flags[idx]:  # Update the refined response only for active examples
                    input[f"self-refine_response-[turn_{t}]"] = refined_batch_outputs[refine_idx].strip("\n#").strip()
                    input[f"self-refine_response-[turn_{t}]_prompt"] = refined_batch_prompts[refine_idx]
                    input["self-refine_response-[final]"] = input[f"self-refine_response-[turn_{t}]"]
                    refine_idx += 1

        for final_input in chunk:
            with open(FLAGS.output_path, "a") as f:
                f.write(json.dumps(final_input) + "\n")

if __name__ == "__main__":
    app.run(main)