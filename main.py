import os 

import streamlit as st
import pandas as pd

from agents.pd_agent import pandas_agent
from utils.eda_functions import eda_functions

st.title("Minerva, Your Personal Data Science Assistant")
st.write("Hello ! I am Minerva, your personal data science assistant. I am here to help you with your data science tasks. Let's get started !")

with st.sidebar:
    st.caption('''
            Minerva is powered by the Langchain AI engine and Google's Generative AI. 
            Minerva can help you with data cleaning, data visualization, and exploratory data analysis.
            To get started, select the type of data you want to analyze and upload a file.
            Minerva will analyze the data and provide you with insights and recommendations.
            You can also ask Minerva questions about the data and get answers in natural language.
            ''')
    
    st.divider()

    st.caption("<p style = 'text-align:center'> Made by Yassine Nefzi </p>", unsafe_allow_html=True)
    st.caption("<p style = 'text-align:center'> Contact : ynyassine7@gmail.com </p>", unsafe_allow_html=True)


if 'clicked' not in st.session_state:
    st.session_state.clicked = {1:False}

def clicked(button):
    st.session_state.clicked[button] = True

st.button("Get Started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    st.header("Exploratory Data Analysis")
    st.subheader("Solution")

    csv = st.file_uploader("Upload your CSV file here", type=["csv"])

    if csv is not None:
        csv.seek(0)
        df = pd.read_csv(csv, low_memory=False)
    
        pandas_agent = pandas_agent(df)

        eda_functions(df, pandas_agent)
    
    