from pprint import pprint
from typing import Dict, Any
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()
tavily_client = TavilyClient()


@tool
def web_search(query: str) -> Dict[str, Any]:
    """Search the web for information"""
    return tavily_client.search(query)


def main():
    agent = create_agent(
        model="gpt-5-nano",
        tools=[web_search]
    )
    question = HumanMessage(content="Who is the current major of San Francisco?")
    response = agent.invoke({"messages": [question]})
    pprint(response["messages"])


if __name__ == '__main__':
    main()