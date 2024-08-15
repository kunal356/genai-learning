import os
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

# Loading Environment variables
load_dotenv()

# Langchain Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'

# Creating Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistance. Please answer the question according to your knowledge"),
        ("user", "Question: {question}")
    ]
)


# Function for getting response from LLM
def get_response(question):
    llm = Ollama(model='gemma2:2b')
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke({"question": question})
    return response


st.title('Q and A chatbot')
st.sidebar.title('Settings')

question = st.text_input("Enter your question here")
if question:
    answer = get_response(question=question)
    st.write(answer)
