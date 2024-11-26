import streamlit as st 
from langchain_core.prompts import ChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_core.output_parsers import StrOutputParser
import numpy as np


import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

# promt=ChatMessagePromptTemplate.from_template(
#     [
#         ("system","You are an helpful assistant respond to the questions to the best of your ability"),
#         ("user","question: {Question}")
#     ]
# )

from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate([
    ("system", "You are a helpful AI bot. Your name is Carl."),
    ("human", "question: {Question}"),
])




parser=StrOutputParser()

def genrate_response(question,api_key,llm,temperature,max_tokens):
    genai.configure(api_key=api_key)
    llm=ChatGoogleGenerativeAI(model=llm)
    chain= template|llm|parser
    answer=chain.invoke({"Question":question})
    return answer 

st.set_page_config("CHAT BOT")
st.title("Q&A ChatBot Using GOOGLE models")

options = np.arange(0.0, 1.1, 0.1)
options1= range(30,601)

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your api_key",type="password")
llm=st.sidebar.selectbox("Choose Model",["gemini-1.5-flash","gemini-1.5-pro"])
temperature=st.sidebar.select_slider("Select Temperature",options=options,value=0.7)
max_tokens=st.sidebar.select_slider("Max Token",options=options1,value=300)

st.chat_message("assistant").write("How can I help you today?")
user_input=st.chat_input(placeholder="What is Machine Learning?")



if user_input and api_key:
    st.chat_message("user").write(user_input)
    response=genrate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
