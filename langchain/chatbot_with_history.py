import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

groq_api_key = os.getenv("GROQ_API_KEY")
print("")
model = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

print(model)

response = model.invoke([HumanMessage(content="My name is bharat and i live in pune")])
print(response)

model.invoke(
    [
        HumanMessage(content="My name is bharat and i live in pune"),
        AIMessage(content="Namaste Bharat, it's nice to meet you. Pune is a wonderful city, known for its rich cultural heritage, educational institutions, and vibrant atmosphere. What do you like to do in your free time, Bharat? Are you a student or working professional?"),
        HumanMessage(content="what is my name?")
        ]
    )