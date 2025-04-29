from dotenv import load_dotenv
from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    DecodingMethod,
    HumanMessage,
    SystemMessage,
    TextGenerationParameters,
    TextGenerationReturnOptions,
    LengthPenalty,
)
from genai.text.generation import CreateExecutionOptions

## ---- Load Credentials ---
import os
#load_dotenv()
my_key = os.getenv('GENAI_KEY')

client = Client(credentials=Credentials(
    api_key = my_key,
    api_endpoint = "https://bam-api.res.ibm.com"))


class bam_chat_agent:
    def __init__(
        self,
        model_name: str = 'mistralai/Mixtral-8x22B-Instruct-v0.1',
        max_new_tokens: int = 2048,
        min_new_tokens:int = 30,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: int = 3407,
        length_penalty_start_index: int = 256,
        length_penalty_decay_factor: float = 1.1,
    ) -> None:
        self.model_name = model_name
        ## ---- Specify Model Parameters ---
        self.parameters = TextGenerationParameters(
            decoding_method=DecodingMethod.SAMPLE,
            length_penalty=LengthPenalty(
                start_index=length_penalty_start_index,
                decay_factor=length_penalty_decay_factor,
            ),
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            # random_seed=seed,
            return_options=TextGenerationReturnOptions(input_text=True),
        )

    def chat(self, system_message, user_message) -> str:
        if system_message == None:
            messages = [HumanMessage(content=user_message)]
        else:
            messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=user_message),
            ]

        response = client.text.chat.create(
            model_id = self.model_name,
            parameters=self.parameters,
            messages = messages,
        )

        print (response.results[0].input_text) # to check the "system" & "user" instruction formats provided to the LLM
        return response.results[0].generated_text

    def format_input(self, system_prompt, user_prompt):
        if "mixtral" in self.model_name:
            return f"<s>[INST] {system_prompt} [/INST]</s> [INST] {user_prompt} [/INST]</s>"
        elif "codellama" in self.model_name:
            return f"""[INST] <<SYS>>{system_prompt}<</SYS>>\n\n{user_prompt} [/INST]"""
        elif "deepseek-coder" in self.model_name:
            return f"""{system_prompt}\n### Instruction:\n{user_prompt}\n### Response:"""
        elif "llama-2" in self.model_name:
            return f"""<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"""
        elif "llama-3" in self.model_name:
            return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
        elif "merlinite" in self.model_name:
            if system_prompt == None:
                return f"""<|user|>\n{user_prompt}\n<|assistant|>"""
            else:
                return f"""<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>"""
        elif "prometheus" in self.model_name:
            if system_prompt == None:
                return f"User: {user_prompt} \n\nAssistant: "
            else:
                return f"{system_prompt}\n\nUser: {user_prompt} \n\nAssistant: "
        else:
            raise ValueError(f"Model {self.model_name} not recognized!")

    def format_input_prometheus(self, system_prompt, user_prompt):
        """ Input format for prometheus when using batch_generate """
        if system_prompt == None:
            return f"User: {user_prompt} \n\nAssistant: "
        else:
            return f"{system_prompt}\n\nUser: {user_prompt} \n\nAssistant: "

    def batch_generate(self, batch_prompts):
        """ Here each prompt in batch_prompts should be formatted using the format_input functions """
        assert type(batch_prompts) == list
        cleaned_responses = []

        for _, response in enumerate(
            client.text.generation.create(
                model_id=self.model_name,
                inputs=batch_prompts,
                execution_options=CreateExecutionOptions(ordered=True),
                parameters=self.parameters)
        ):
            [result] = response.results
            cleaned_result = result.generated_text
            cleaned_responses.append(cleaned_result)
        return cleaned_responses

if __name__ == "__main__":
    model_id = "kaist-ai/prometheus-8x7b-v2" # "mistralai/mixtral-8x7b-instruct-v01" # "kaist-ai/prometheus-8x7b-v2"
    llm_agent = bam_chat_agent(model_id)

    from termcolor import colored
    print (colored(llm_agent.chat(system_message="", user_message="Write a story."), "green"))