import os

import streamlit as st
import pandas as pd

from agents.pd_agent import pandas_agent
from agents.agent_functions import answer_user_query
from utils.eda_functions import eda_functions, variable_query
from constants import MINERVA_DESCRIPTION

st.title("Minerva, Your Personal Data Science Assistant")
st.write(
    "Hello ! I am Minerva, your personal data science assistant. I am here to help you with your data science tasks. Let's get started !"
)

with st.sidebar:
    st.subheader("About Minerva")

    st.caption(MINERVA_DESCRIPTION)

    st.divider()

    st.caption(
        "<p style = 'text-align:center'> Made by Yassine Nefzi </p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "<p style = 'text-align:center'> Contact : ynyassine7@gmail.com </p>",
        unsafe_allow_html=True,
    )


if "clicked" not in st.session_state:
    st.session_state.clicked = {1: False}


def clicked(button):
    st.session_state.clicked[button] = True


st.button("Get Started", on_click=clicked, args=[1])

if st.session_state.clicked[1]:
    st.header("Upload Your Dataset")

    csv = st.file_uploader("Upload your CSV file here", type=["csv"])

    if csv is not None:
        csv.seek(0)
        df = pd.read_csv(csv, low_memory=False)

        pandas_agent = pandas_agent(df)

        st.header("Exploratory Data Analysis")

        eda_functions(df, pandas_agent)

        st.divider()

        user_column_query = st.text_input(
            "Select a column to analyze (make sure to use the exact column name)"
        )
        st.write("OR")
        user_general_query = st.text_input(
            "Ask any general questions about the dataset"
        )

        if user_column_query and not user_general_query:
            variable_query(df, pandas_agent, user_column_query)

        elif user_general_query and not user_column_query:
            answer = answer_user_query(df, pandas_agent, user_general_query)
            st.write(answer)
