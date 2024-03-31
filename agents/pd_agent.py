from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

from utils.llm import get_llm

llm = get_llm()

def pandas_agent(df):
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
    )

# df = pd.read_csv(
#     "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"
# )

# agent = pandas_agent(df)

# agent.invoke("what is the size of the dataset")



