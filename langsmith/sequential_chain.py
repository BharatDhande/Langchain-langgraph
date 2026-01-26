from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Sequential LLM App'

prompt_1 = PromptTemplate(
    template = 'Generate a detailed report on topic {topic}',
    input_variables=['topic']
    )

prompt_2 = PromptTemplate(
    template = 'Generate 5 pointer summery from text \n {text}',
    input_variables=['text']
    )

model = ChatOpenAI(
        model="xiaomi/mimo-v2-flash:free",
        api_key=os.environ.get('OPEN_ROUTER_API_KEY'),
        base_url="https://openrouter.ai/api/v1",
    )

parser = StrOutputParser()

chain = prompt_1 | model | parser | prompt_2 | model | parser

config = {
    'run_name': 'Sequential chain',
    'tags': ['llm app', 'report generation', 'summerization'],
    'metadata': {'model': 'xiaomi/mimo-v2-flash:free'}
    }

result = chain.invoke({'topic': 'first religion in world'}, config=config)

print(result)