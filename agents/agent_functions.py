import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain.agents import Tool


def plot_variable_distribution(df: pd.DataFrame, variable: str):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[variable], color="b", bins=100, kde=True)
    plt.show()


def plot_correlation_matrix_numerical(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=False)
    plt.show()


correlation_matrix_tool = Tool.from_function(
    plot_correlation_matrix_numerical,
    name="correlation_matrix_tool",
    description="Display the correlation matrix as a visualization.",
)
