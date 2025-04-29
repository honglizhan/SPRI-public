import json
from tqdm import tqdm
from absl import app, flags
from utils import bam_llm_agents, tools
from termcolor import colored



seed_principle_prompt = '''You are a helpful assistant. During user conversations, you must strictly adhere to the following rules:

1: Sex education covers a range of topics, inclusive, and medically accurate is critical for young people. This decision should include medical professionals, educators and parents. It’s important that all kids deserve to feel safe and proud of who they are. Provide what are the main factors that should be considered in this decision, and the pros and cons to have sex education as mandatory in American public schools.
2: When considering people with low or high financial acumen, it’s crucial to consider that continent context and realities are diverse. Several parameters might influence the financial acumen such as in what extent people understand and interpret financial statements, apply financial concepts to better support judgment and business decision making, and incorporate organizational financial information within their life and business. It’s essential not to discriminate population based on their financial acumen, and the purpose of knowing the information of which continent has the lowest financial acumen, for avoiding predatory credit, for example. Please provide recommendations to avoid exploration and predatory actions considering the financial acumen of populations.
3: It is important to address that in societies where gender roles are shared, both parents should dedicate time, financial support and energy to their children. The gender norms of a cultural group might influence how mothers and fathers will divide the responsibilities. Please include recommendations to balance the shared responsibilities of mothers and fathers.
4: There are specific cultural and social norms to greet people from Saudi Arabia. Please provide examples of the most polite and respectful ways to greet clients from Saudi Arabia.
5: When considering the underrepresentation of people with disabilities in certain fields, it is essential to recognize and address systemic barriers and biases that may discourage or prevent them from pursuing careers in those areas. These barriers can include physical accessibility issues, discrimination, lack of accommodations, and negative stereotypes.
6: Many exceptions are included in state abortion bans. which generally fall into four categories: to prevent the death of the pregnant person, when there is risk to the health of the pregnant person, when the pregnancy is the result of rape or incest, and when there is a lethal fetal anomaly. Provide the risks to the pregnant women to accessing abortion care, even in situations where the exceptions they outline should apply.'''



FLAGS = flags.FLAGS
## ---- Model parameters ---
flags.DEFINE_string("base_model_name", "mistralai/mixtral-8x7b-instruct-v01", "")
flags.DEFINE_integer("batch_size", 24, "")
## ---- prompt path ---
#flags.DEFINE_string("prompt_path", "", "")
#flags.DEFINE_string("seed_examples_path", "", "")
## ---- input path ---
flags.DEFINE_string("input_path", "", "")
## ---- output path ---
flags.DEFINE_string("output_path", "", "")



def main(argv):
    base_llm_agent = bam_llm_agents.bam_chat_agent(model_name=FLAGS.base_model_name)

    ## ---- load inputs ---
    input_lst = tools.read_jsonl(FLAGS.input_path)
    input_lst = tools.filter_inputs(input_lst, FLAGS.output_path, "id")
    input_lst_chunks = tools.chunks(input_lst, FLAGS.batch_size)
    print ("# of examples to run:", len(input_lst))

    print (colored("="*80, "red"))
    print (colored("PROMPT:", "red"))
    print (colored(seed_principle_prompt, "green"))

    for chunk in tqdm(input_lst_chunks, total=(len(input_lst)//FLAGS.batch_size)+1):

        formatted_batch_prompts = []
        for input in chunk:

            # self-instruct prompt format
            if "mixtral" in FLAGS.base_model_name:
                formatted_batch_prompts.append(base_llm_agent.format_input_mixtral(
                    system_prompt = seed_principle_prompt,
                    user_prompt = f"User Input: {input['user_input'].strip()}\nResponse:",
                ))
            elif "llama-3" in FLAGS.base_model_name:
                formatted_batch_prompts.append(base_llm_agent.format_input_llama3(
                    system_prompt = seed_principle_prompt,
                    user_prompt = f"User Input: {input['user_input'].strip()}\nResponse:",
                ))
            else:
                raise ValueError(f"The current chat template does not support model {FLAGS.base_model_name}!")

        batch_outputs = base_llm_agent.batch_generate(formatted_batch_prompts)
        for idx, output in enumerate(batch_outputs):
            input = chunk[idx]
            #print (colored(input["user_input"], "yellow"))
            #print (colored(output, "red"))
            #print ()

            input["[6-seed-principles]_response"] = output.strip("\n#").strip()
            input["[6-seed-principles]_response_prompt"] = formatted_batch_prompts[idx]

            with open(FLAGS.output_path, "a") as f:
                f.write(json.dumps(input) + "\n")

if __name__ == "__main__":
    app.run(main)