from genai.client import Client
from genai.credentials import Credentials
from genai.schema import (
    DecodingMethod,
    HumanMessage,
    AIMessage,
    SystemMessage,
    TextGenerationParameters,
    TextGenerationReturnOptions,
    LengthPenalty,
)
import pandas as pd

## ---- Load Credentials ---
import os
my_key = os.getenv('GENAI_KEY')

client = Client(credentials=Credentials(
    api_key = my_key,
    api_endpoint = "https://bam-api.res.ibm.com"))


class bam_chat_agent:
    def __init__(
        self,
        model_name: str = 'mistralai/Mixtral-8x22B-Instruct-v0.1',
        max_new_tokens: int = 512,
        min_new_tokens:int = 30,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        seed: int = 42,
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
            random_seed=seed,
            return_options=TextGenerationReturnOptions(input_text=True),
        )
        self.chat_history = []  # Initialize chat history

    def chat(self, system_message, user_message) -> str:
        if system_message is not None:
            self.chat_history.append(SystemMessage(content=system_message))

        self.chat_history.append(HumanMessage(content=user_message))

        response = client.text.chat.create(
            model_id = self.model_name,
            parameters = self.parameters,
            messages = self.chat_history,
        )

        # print (response.results[0].input_text) # to check the "system" & "user" instruction formats provided to the LLM
        self.chat_history.append(AIMessage(content=response.results[0].generated_text))
        return response.results[0].input_text, response.results[0].generated_text

    def reset(self):
        # Clear the chat history
        self.chat_history = []
        print("Chat history has been reset.")


class openai_chat_agent:
    def __init__(
        self,
        model_name,
        client,
    ) -> None:
        self.model_name = model_name
        self.client = client
        self.chat_history = []

    def reset(self) -> None:
        self.chat_history = []

    def chat(self, system_message, user_message: str) -> str:
        if system_message is not None:
            self.chat_history.append({"role": "system", "content": system_message})
        self.chat_history.append({"role": "user", "content": user_message})

        ai_message = self.client.chat.completions.create(
            model = self.model_name,
            messages = self.chat_history,
            temperature = 0.7,
            max_tokens = 512,
            seed = 42,
        ).choices[0].message.content

        self.chat_history.append({"role": "assistant", "content": ai_message})
        return self.chat_history, ai_message


if __name__ == "__main__":
    model_id = "meta-llama/llama-3-70b-instruct" # "mistralai/mixtral-8x7b-instruct-v01" # "kaist-ai/prometheus-8x7b-v2"
    llm_agent = bam_chat_agent(model_id)
    print (llm_agent.chat(system_message="You are a chatbot.", user_message="What is your favorite condiment?"))