import streamlit as st
import pandas as pd

from agents.pd_agent import pandas_agent
import numpy as np

def eda_functions(df: pd.DataFrame, pandas_agent: pandas_agent):
    st.write("**Data Overview**")
    st.write("The first rows of your dataset look like this:")
    st.write(df.head())

    st.write("**The shape of your dataset:**")
    st.write(df.shape)
    
    st.write("The meaning of your columns")
    columns_meaning = pandas_agent.invoke("What is the meaning of the columns")
    st.caption(columns_meaning.get("output"))

    st.write("The data types of your columns")
    columns_types = pandas_agent.invoke("What are the data types of the columns")
    st.caption(columns_types.get("output"))

    st.write("Columns containing missing values in your dataset")
    missing_values = pandas_agent.invoke("What are the columns containing missing values in the dataset")
    st.caption(missing_values.get("output"))

    st.write("The number of duplicates in your dataset")
    duplicates = pandas_agent.invoke("What are the duplicates in the dataset")
    st.caption(duplicates.get("output"))

    
    st.write("**Data Summary**")
    st.write("The summary statistics of your dataset")
    st.write(df.describe())

    # st.write("The correlation between the columns in your dataset")
    # correlation = pandas_agent.invoke("what are the most correlated columns in the dataset")
    # st.write(correlation.get("output"))

    st.write("New features that might be useful to add")
    new_features = pandas_agent.invoke("what are some new features that might be useful to add")
    st.caption(new_features.get("output"))

