from dataclasses import dataclass
from typing import Dict, Any, Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelResponse, ModelRequest
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient()

db = SQLDatabase.from_uri("sqlite:///resources/Chinook.db")

@dataclass
class UserRole:
    """Custom role schema for user authentication."""
    user_role: str = "external"

@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web using Tavily client."""
    return tavily_client.search(query)

@tool
def sql_query(query: str) -> str:
    """Obtain information from the database using SQL queries"""

    try:
        return db.run(query)
    except Exception as e:
        return f"Error: {e}"


@wrap_model_call
def dynamic_tool_call(request: ModelRequest,
handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    """Dynamically call tools based on the runtime context"""

    user_role = request.runtime.context.user_role
    if user_role == "internal":
        pass
    else:
        tools = [web_search]
        request = request.override(tools=tools)

    return handler(request)

def init_agent():
    return create_agent(
        model="gpt-5-nano",
        tools=[web_search, sql_query],
        middleware=[dynamic_tool_call],
        context_schema=UserRole
    )


def main():
    response = init_agent().invoke(
        {"messages": [HumanMessage(content="How many artists are in the database?")]},
        context={"user_role": "external"}
    )

    print(response["messages"][-1].content)


if __name__ == '__main__':
    main()