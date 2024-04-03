from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.tools import PythonREPLTool

from utils.llm import get_llm
from .agent_functions import correlation_matrix_tool


llm = get_llm()

tools = [PythonREPLTool()]


def pandas_agent(df):
    return create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        extra_tools=tools,
    )
