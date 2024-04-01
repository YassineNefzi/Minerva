import streamlit as st
import pandas as pd

from agents.pd_agent import pandas_agent
import numpy as np


@st.cache_data
def eda_functions(df: pd.DataFrame, _pandas_agent: pandas_agent):
    st.subheader("Data Overview", divider="blue")
    st.subheader("First Rows")
    st.write(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Columns Meaning")
    columns_meaning = _pandas_agent.invoke("What is the meaning of the columns")
    st.write(columns_meaning.get("output"))

    st.subheader("Columns Types")
    columns_types = _pandas_agent.invoke("What are the data types of the columns")
    st.write(columns_types.get("output"))

    st.subheader("Missing Values")
    missing_values = _pandas_agent.invoke(
        "What are the names of the columns containing missing values in the dataset"
    )
    st.write(missing_values.get("output"))

    st.subheader("Duplicates")
    duplicates = _pandas_agent.invoke("What are the duplicates in the dataset")
    st.write(duplicates.get("output"))

    st.subheader("Data Summary", divider="blue")
    st.subheader("Statistics")
    st.write(df.describe())

    st.subheader("Correlation Analysis")
    correlation = _pandas_agent.invoke(
        "what are the most correlated columns in the dataset ? do not use np.corr() or any external library to calculate the correlation."
    )
    st.write(correlation.get("output"))

    st.subheader("Potential New Features")
    new_features = _pandas_agent.invoke(
        "what are some new features that might be useful to add"
    )
    st.write(new_features.get("output"))


@st.cache_data
def variable_query(df: pd.DataFrame, _pandas_agent: pandas_agent, query: str):
    st.header(f"**More in depth analysis of the {query} column**", divider="blue")
    st.subheader("Line Chart")
    st.line_chart(df, y=[query])

    st.subheader(f"Summary Statistics {query}")
    summary_statistics = _pandas_agent.invoke(
        f"Give me a summary of the statistics of the {query} column"
    )
    st.write(summary_statistics.get("output"))

    st.subheader(f"Missing Values in {query}")
    missing_values = _pandas_agent.invoke(
        f"How many missing values are there in the {query} column"
    )
    st.write(missing_values.get("output"))

    st.subheader(f"Normality of {query}")
    normality = _pandas_agent.invoke(
        f"""Check for normality or specific distribution shapes of {query}. If possible try to plot its 
        distribution but do not use any external libraries."""
    )
    st.write(normality.get("output"))

    st.subheader(f"Outliers in the {query} column")
    outliers = _pandas_agent.invoke(
        f"Assess the presence of outliers of {query} column"
    )
    st.write(outliers.get("output"))

    st.subheader(f"Correlation Between {query} and Other Columns")
    correlation = _pandas_agent.invoke(
        f"""What are the names of the most correlated columns with the {query} column and the percentage of correlation. Do not use 
        numpy or np.corr() or any external library to calculate the correlation. Calculate the correlation only if the column is numerical.
        In case the column is categorical, output this : Correlation for a categorical feature is not supported at the moment."""
    )
    st.write(correlation.get("output"))

    st.subheader(f"Relation Between {query} and Target")
    relation = _pandas_agent.invoke(
        f"What is the relationship between the {query} column and the target variable"
    )
    st.write(relation.get("output"))

    st.subheader(f"Trends, Seasonality, and Cyclic Patterns of {query}")
    trends = _pandas_agent.invoke(
        f"Analyse trends, seasonality, and cyclic patterns of {query}"
    )
    st.write(trends.get("output"))
