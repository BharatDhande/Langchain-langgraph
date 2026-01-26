from langsmith import Client
import os
from dotenv import load_dotenv

load_dotenv()

print("API Key Loaded:", os.getenv("LANGCHAIN_API_KEY") is not None)
print("Tracing Enabled:", os.getenv("LANGCHAIN_TRACING_V2"))
print("Project:", os.getenv("LANGCHAIN_PROJECT"))

client = Client()
print("Your LangSmith Projects:")
for proj in client.list_projects():
    print("-", proj.name)
