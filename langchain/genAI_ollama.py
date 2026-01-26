from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tracers.context import tracing_v2_enabled
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"

st.title("Langchain Demo With Gemma Model")

st.write("LangSmith connected?", os.getenv("LANGCHAIN_API_KEY") is not None)
st.write("Project name:", os.getenv("LANGCHAIN_PROJECT"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "give ans only of you know correct ans dont give correct madepu ans."),
    ("user", "Question: {question}")
])

input_text = st.text_input("What question do you have in mind?")

llm = Ollama(model="gemma:2b")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    with tracing_v2_enabled():
        response = chain.invoke({"question": input_text})
    st.write(response)
