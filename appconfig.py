import os
from dotenv import load_dotenv

load_dotenv()


class EnvConfig:

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.default_model_name = os.getenv("DEFAULT_MODEL_NAME")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")


# Instantiate env config
env_config = EnvConfig()
