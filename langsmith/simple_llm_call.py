from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

prompt = PromptTemplate.from_template("{question}")

model = ChatOpenAI(
        model="xiaomi/mimo-v2-flash:free",
        api_key=os.environ.get('OPEN_ROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1",
    )

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'question':"indias highest AQI recorded?"})

print(result)