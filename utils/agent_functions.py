import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


def plot_variable_distribution(df: pd.DataFrame, variable: str):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[variable], color="b", bins=100, kde=True)
    plt.show()


def plot_correlation_matrix_numerical(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=False)
    plt.show()
