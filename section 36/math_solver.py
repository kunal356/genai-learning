import streamlit as st

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain, LLMChain


st.set_page_config(
    page_title="Text To MAth Problem Solver And Data Serach Assistant", page_icon="🧮")
st.title("Text To Math Problem Solver Uing Google Gemma 2")

api_key = st.sidebar.text_input('Enter your Groq Api key', type='password')


if not api_key:
    st.info("Please add your Groq API key to continue")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model='Gemma2-9b-It')

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki_tool = Tool(
    name='Wikipedia',
    func=wiki_wrapper.run,
    description='A tool for searching the Internet to find the vatious information on the topics mentioned'
)

math_chain = LLMMathChain.from_llm(llm=llm)

calculator = Tool(
    name='Calculator',
    func=math_chain.run,
    description='A tools for answering math related questions. Only input mathematical expression need to bed provided'
)

prompt = """
You are a agent tasked for solving users mathemtical question. Logically arrive at the solution and provide a detailed explanation
and display it point wise for the question below
Question:{question}
Answer:
"""
prompt_template = PromptTemplate(template=prompt,
                                 input_variables=['question']
                                 )
chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning tool",
    func=chain.run,
    description='A tool for answering logic-based and reasoning questions.'
)

assistant_agent = initialize_agent(
    tools=[wiki_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I am Math chatbot. How can I help you?"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.text_area("Enter youe question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

if st.button("Find Answer"):
    if question:
        with st.spinner("Generating response.."):
            st.session_state["messages"].append({"role": "user", "content": question})
            st.chat_message("user").write(question)

            st_cb =StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages, callbacks=[st_cb])
            st.session_state["messages"].append({'role':'assistant', 'content': response})
            st.success(response)
    else:
        st.warning("Please enter the question")