from langchain_community.llms import Ollama
import os
from dotenv import load_dotenv

from langchain_community.llms import ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question:{question}")
    ]
)

# streamlit framework
st.title("Langchain Demo with Gemma Model")
input_text = st.text_input("What question you have in mind?")

# Using Gemma 2 LLM
llm = Ollama(model="gemma2:2b")

# Creating a output parser to format the output
ouptut_parser = StrOutputParser()

# creating a chain
chain = prompt | llm | ouptut_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))