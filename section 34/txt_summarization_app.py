import streamlit as st
import validators
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain

load_dotenv()

st.set_page_config(page_title='LangChain: Summarize Text From YT or Website',page_icon='🦜')
st.title('🦜 LangChain: Summarize Text From YT or Website')
st.subheader('Summarize URL')

with st.sidebar:
    grop_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=grop_api_key)

prompt_template = """
Provide a summary of the following content in 300 words:
Content:{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['text'])


if st.button("Summarize Content"):
    if not grop_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid url. It can be a YT url or any website url.")
    else:
        try:
            with st.spinner("Waiting..."):
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                st.write(docs)
                output_summary=chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception: {e}")