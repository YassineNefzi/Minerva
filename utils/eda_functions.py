import streamlit as st
import pandas as pd
import numpy as np

from agents.pd_agent import pandas_agent


@st.cache_data
def eda_functions(df: pd.DataFrame, _pandas_agent: pandas_agent):
    st.subheader("Data Overview", divider="blue")
    st.subheader("First Rows")
    st.write(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Columns Meaning")
    columns_meaning = _pandas_agent.invoke(
        """Give me the meaning of each column in the dataset in the following format :
        | Column Name | Meaning |
        |-------------|---------|
        | column1     | meaning1|
        | column2     | meaning2|
        The meaning refers to explaining what the column represents in the dataset. example : column1 : age of the person.
        Note that df is the dataframe you are working with."""
    )
    st.write(columns_meaning.get("output"))

    st.subheader("Columns Types")
    columns_types = _pandas_agent.invoke(
        """Give me the data types of each column in the dataset in the following format :
        | Column Name | Data Type |
        |-------------|-----------|
        | column1     | int64     |
        | column2     | object    |
        Replace the column names and data types with the actual column names and data types of the dataset. example : column1 : int64.
        Note that df is the dataframe you are working with."""
    )
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

    st.subheader("Correlation Matrix")
    correlation = _pandas_agent.invoke(
        """Find the correlation matrix between the different numerical features of the dataset.
        Your output should look like this and do not forget the | and - characters to make the table look nice :
        |        | column1 | column2 | column3 | column4 |
        |--------|---------|---------|---------|---------|
        | column1| 1.0     | 0.5     | 0.7     | 0.3     |
        | column2| 0.5     | 1.0     | 0.6     | 0.2     |
        | column3| 0.7     | 0.6     | 1.0     | 0.1     |
        | column4| 0.3     | 0.2     | 0.1     | 1.0     |
        Of course replace the column names with the actual column names of your dataset and the percentages also.
        You may use any tools at your disposal and import all the necessary libraries.
        df is the dataframe you are currently working with."""
    )
    st.write(correlation.get("output"))

    st.subheader("Potential New Features")
    new_features = _pandas_agent.invoke(
        "what are some new features that might be useful to add. df is the dataframe you are working with."
    )
    st.write(new_features.get("output"))


@st.cache_data
def variable_query(df: pd.DataFrame, _pandas_agent: pandas_agent, query: str):
    st.header(f"**More in depth analysis of the {query} column**", divider="blue")
    st.subheader("Line Chart")
    st.line_chart(df, y=[query])

    st.subheader(f"Summary Statistics of {query}")
    summary_statistics = _pandas_agent.invoke(
        f"Give me a summary of the statistics of the {query} column. Note that df is the dataframe you are working with."
    )
    st.write(summary_statistics.get("output"))

    st.subheader(f"Missing Values in {query}")
    missing_values = _pandas_agent.invoke(
        f"How many missing values are there in the {query} column ? Note that df is the dataframe you are working with."
    )
    st.write(missing_values.get("output"))

    st.subheader(f"Normality of {query}")
    normality = _pandas_agent.invoke(
        f"""Check for normality or specific distribution shapes of {query}. Note that df is the dataframe you are working with."""
    )
    st.write(normality.get("output"))

    st.subheader(f"Outliers in the {query} column")
    outliers = _pandas_agent.invoke(
        f"Assess the presence of outliers of {query} column. Note that df is the dataframe you are working with."
    )
    st.write(outliers.get("output"))

    st.subheader(f"Relation Between {query} and Target")
    relation = _pandas_agent.invoke(
        f"What is the relationship between the {query} column and the target variable ? Note that df is the dataframe you are working with."
    )
    st.write(relation.get("output"))
