import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from .pd_agent import pandas_agent


@st.cache_data
def answer_user_query(df: pd.DataFrame, _pandas_agent: pandas_agent, query: str):
    answer = _pandas_agent.invoke(
        f"""Provide a response for this query : 
        {query}
        Your response has to be a string.
        You may use any tools at your disposal and import all the necessary libraries. 
        Note that df is the dataframe you are working with."""
    )
    return answer.get("output")
