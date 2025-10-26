from appconfig import env_config
from groq import Groq
from openai import OpenAI


class GroqModel:
    def __init__(self, api_key=env_config.groq_api_key, model_name="llama-3.3-70b-versatile"):
        if not api_key:
            raise ValueError("No API key provided")

        self.api_key = api_key
        self.model_name = model_name
        self.client = Groq(api_key=self.api_key)
        self.conversation_history = []
        self.chatbot_name = "Chip"

    def clear_history(self):
        if self.conversation_history:
            self.conversation_history = []

    def chat(self, user_message, system_prompt=None, temperature=0.5, max_tokens=1024):
        messages = []

        # Setup chatbot name and optional user system prompt
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": f"You are {self.chatbot_name}, {system_prompt}",
                }
            )
        else:
            messages.append(
                {
                    "role": "system",
                    "content": f"You are {self.chatbot_name}, a helpful and friendly AI assistant.",
                }
            )

        # add conversation history
        if self.conversation_history:
            messages.extend(self.conversation_history)

        # add user message
        messages.append(
            {
                "role": "user",
                "content": f"{user_message}",
            }
        )

        # make call to Groq API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # cache message and response
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        self.conversation_history.append(response.choices[0].message)

        return response.choices[0].message.content


class OpenAIModel:
    def __init__(self, api_key=env_config.openai_api_key, model_name="gpt-5"):
        if not api_key:
            raise ValueError("No API key provided")

        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=self.api_key)
        self.conversation_history = []
        self.chatbot_name = "Alfred"

    def chat(self, user_message, system_prompt=None, temperature=0.5, max_tokens=1024):
        self.conversation_history.append({
            "role": "user",
            "content": f"{user_message}"
        })

        # leaving out temperature and max_tokens as they are not supported by gpt-5 models
        response = self.client.responses.create(
            model=self.model_name,
            input=self.conversation_history,
            instructions=f"{system_prompt}" if system_prompt else f"You are {self.chatbot_name}, a helpful and friendly AI assistant.",
        )

        # Add the response to the conversation history
        self.conversation_history += [{"role": el.role, "content": el.content} for el in response.output if hasattr(el, "role")]

        return response.output_text

    def clear_history(self):
        if self.conversation_history:
            self.conversation_history = []


def llm_factory(model_name="llama-3.3-70b-versatile", api_key=None):
    llm_classes = {
        "llama-3.1-8b-instant": GroqModel,
        "llama-3.3-70b-versatile": GroqModel,
        "gpt-5": OpenAIModel,
        "gpt-5-mini": OpenAIModel,
        "gpt-5-nano": OpenAIModel
    }
    params = {}

    if not api_key:
        print("No API key provided, will use the default one for the selected model.")
    else:
        params["api_key"] = api_key

    if model_name not in llm_classes:
        raise ValueError("You need to specify a valid model")
    else:
        print(f"Model to use: {model_name}")
        params["model_name"] = model_name

    return llm_classes[model_name](**params)


if __name__ == "__main__":
    app = llm_factory(model_name="gpt-5-mini")
    while True:
        message = input(f"What would you like to ask: ")
        response = app.chat(message)
        print(f"\nAssistant response: {response}\n")
