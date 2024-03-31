import os 

import streamlit as st


st.title("Minerva, Your Personal Data Science Assistant")
st.write("Hello ! I am Minerva, your personal data science assistant. I am here to help you with your data science tasks. Let's get started !")
st.header("Exploratory Data Analysis")
st.subheader("Solution")

with st.sidebar:
    st.write('''
            Minerva is a data science assistant that helps you with your data science tasks. Minerva is powered by the Langchain AI engine and Google's Generative AI. 
            Minerva can help you with data cleaning, data visualization, and exploratory data analysis.
            To get started, select the type of data you want to analyze and upload a file.
            Minerva will analyze the data and provide you with insights and recommendations.
            You can also ask Minerva questions about the data and get answers in natural language
            ''')