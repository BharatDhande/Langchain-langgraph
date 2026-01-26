from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv
load_dotenv()


api_key = os.getenv("GROQ_API_KEY")
print("Loaded KEy: ",api_key)
model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=api_key)

#create prompt template
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages([
    ("system",system_template),
    ("user", '{text}')
    ])

parser = StrOutputParser()

chain = prompt_template|model|parser

app=FastAPI(title="Langchain server",
            version="1.0",
            description="A simple API server")


add_routes(
    app,
    chain,
    path="/chain"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8088)