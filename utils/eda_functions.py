import streamlit as st
import pandas as pd

from agents.pd_agent import pandas_agent
import numpy as np


@st.cache_data
def eda_functions(df: pd.DataFrame, _pandas_agent: pandas_agent):
    st.write("**Data Overview**")
    st.write("The first rows of your dataset look like this:")
    st.write(df.head())

    st.write("**The shape of your dataset:**")
    st.write(df.shape)

    st.write("The meaning of your columns")
    columns_meaning = _pandas_agent.invoke("What is the meaning of the columns")
    st.caption(columns_meaning.get("output"))

    st.write("The data types of your columns")
    columns_types = _pandas_agent.invoke("What are the data types of the columns")
    st.caption(columns_types.get("output"))

    st.write("Columns containing missing values in your dataset")
    missing_values = _pandas_agent.invoke(
        "What are the names of the columns containing missing values in the dataset"
    )
    st.caption(missing_values.get("output"))

    st.write("The number of duplicates in your dataset")
    duplicates = _pandas_agent.invoke("What are the duplicates in the dataset")
    st.caption(duplicates.get("output"))

    st.write("**Data Summary**")
    st.write("The summary statistics of your dataset")
    st.write(df.describe())

    # st.write("The correlation between the columns in your dataset")
    # correlation = pandas_agent.invoke("what are the most correlated columns in the dataset")
    # st.write(correlation.get("output"))

    st.write("New features that might be useful to add")
    new_features = _pandas_agent.invoke(
        "what are some new features that might be useful to add"
    )
    st.caption(new_features.get("output"))


@st.cache_data
def variable_query(df: pd.DataFrame, _pandas_agent: pandas_agent, query: str):
    st.write(f"**More in depth analysis of the {query} column**")
    st.line_chart(df, y=[query])

    st.write(f"Summary statistics of the {query} column")
    summary_statistics = _pandas_agent.invoke(
        f"Give me a summary of the statistics of the {query} column"
    )
    st.write(summary_statistics.get("output"))

    st.write(f"Missing values in the {query} column")
    missing_values = _pandas_agent.invoke(
        f"How many missing values are there in the {query} column"
    )
    st.write(missing_values.get("output"))

    st.write(f"Normality of the {query} column")
    normality = _pandas_agent.invoke(f"Is the {query} column normally distributed")
    st.write(normality.get("output"))

    st.write(f"Outliers in the {query} column")
    outliers = _pandas_agent.invoke(f"What are the outliers in the {query} column")
    st.write(outliers.get("output"))

    st.write(f"Correlation between the {query} column and the other columns")
    correlation = _pandas_agent.invoke(
        f"What are the names of the most correlated columns with the {query} column and the percentage of correlation"
    )
    st.write(correlation.get("output"))

    st.write(f"Relationship between the {query} column and the target variable")
    relationship = _pandas_agent.invoke(
        f"What is the relationship between the {query} column and the target variable"
    )
    st.write(relationship.get("output"))

    st.write(f"Trends, Seasonality, and Cyclic Patterns of the {query} column")
    trends = _pandas_agent.invoke(
        f"Analyse trends, seasonality, and cyclic patterns of {query}"
    )
    st.write(trends.get("output"))
