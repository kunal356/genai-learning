import os
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

# Loading Environment variables
load_dotenv()

# Langchain Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_PROJECT'] = os.getenv('LANGCHAIN_PROJECT')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Creating Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistance. Please answer the question according to your knowledge"),
        ("user", "Question: {question}")
    ]
)


# Function for getting response from LLM
def get_response(question, model, temperature, maxtoken):
    llm = ChatOpenAI(model=model, api_key=openai_api_key,
                     max_tokens=maxtoken, temperature=temperature)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke({"question": question})
    return response


st.title('Q and A chatbot')
st.sidebar.title('Settings')
model = st.sidebar.selectbox('Select OpenAI model', [
                             'gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo-0125'])
question = st.text_input("Enter your question here")
tempurature = st.sidebar.slider(
    'Temperature', min_value=0.0, max_value=1.0, value=0.7)
maxtoken = st.sidebar.slider(
    'Max token', min_value=0, max_value=300, value=150)
if question:
    answer = get_response(question=question, model=model,
                           temperature=tempurature, maxtoken=maxtoken)
    st.write(answer)
else:
    st.warning('Please provide some query.')
