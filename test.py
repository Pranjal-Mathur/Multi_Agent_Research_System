# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI

# load_dotenv()

# llm = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=0
# )

# print(llm.invoke("Hello"))

import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(".env")

print("ENV exists:", env_path.exists())

load_dotenv(dotenv_path=env_path)

print("GOOGLE:", os.getenv("GOOGLE_API_KEY"))