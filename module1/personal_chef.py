from pprint import pprint
from typing import Dict, Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

load_dotenv()
tavily_client = TavilyClient()

def web_search(query: str) -> Dict[str, Any]:
    """Search the web using Tavily client.

    Args:
        query: The search query string.

    Returns:
        A dictionary containing search results.
    """
    return tavily_client.search(query)

def system_prompt() -> str:
    """Return the system prompt for the personal chef agent.

    Returns:
        A string containing the system prompt instructions.
    """
    return """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

def init_agent() -> Any:
    """Initialize and configure the personal chef agent.

    Returns:
        A configured agent with web search capabilities and memory.
    """
    agent = create_agent(
        model="gpt-5-nano",
        tools=[web_search],
        system_prompt = system_prompt(),
        checkpointer=InMemorySaver()
    )

    return agent

def main():
    """Run the personal chef agent with sample ingredients."""
    config = {"configurable": {"thread_id": "1"}}
    response = init_agent().invoke({"messages": [HumanMessage(content="I have 2 cups of milk and 1 cup of flour")]}, config=config)
    pprint(response["messages"])


if __name__ == '__main__':
    main()