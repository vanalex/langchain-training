"""MCP client integration with LangChain agents."""

import asyncio
from pprint import pprint
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv()

# Constants
SERVER_NAME = "local_server"
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_THREAD_ID = "1"


def create_mcp_client(server_config: dict[str, Any]) -> MultiServerMCPClient:
    """Create and configure an MCP client.

    Args:
        server_config: Dictionary containing server configuration.

    Returns:
        Configured MultiServerMCPClient instance.
    """
    return MultiServerMCPClient(server_config)


async def get_mcp_components(
    client: MultiServerMCPClient, server_name: str
) -> tuple[list[Any], list[Any], str]:
    """Retrieve tools, resources, and prompt from MCP server.

    Args:
        client: The MCP client instance.
        server_name: Name of the server to query.

    Returns:
        Tuple containing (tools, resources, prompt_content).
    """
    tools = await client.get_tools()
    resources = await client.get_resources(server_name)
    prompt_response = await client.get_prompt(server_name, "prompt")
    prompt_content = prompt_response[0].content

    return tools, resources, prompt_content


async def run_agent(
    tools: list[Any],
    system_prompt: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    thread_id: str = DEFAULT_THREAD_ID,
) -> dict[str, Any]:
    """Create and run an agent with the provided configuration.

    Args:
        tools: List of tools available to the agent.
        system_prompt: System prompt for the agent.
        user_message: The user's message to process.
        model: The model identifier to use.
        thread_id: Thread ID for conversation tracking.

    Returns:
        Agent response dictionary.
    """
    agent = create_agent(model=model, tools=tools, system_prompt=system_prompt)

    config = {"configurable": {"thread_id": thread_id}}
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_message)]}, config=config
    )

    return response


async def main() -> None:
    """Run the MCP agent with langchain-mcp-adapters example."""
    # Server configuration
    server_config = {
        SERVER_NAME: {
            "transport": "stdio",
            "command": "python",
            "args": ["resources/mcp_server.py"],
        }
    }

    # Initialize MCP client
    client = create_mcp_client(server_config)

    # Get MCP components
    tools, _resources, prompt = await get_mcp_components(client, SERVER_NAME)

    # Run agent with user query
    response = await run_agent(
        tools=tools,
        system_prompt=prompt,
        user_message="Tell me about the langchain-mcp-adapters library",
    )

    pprint(response)


if __name__ == "__main__":
    asyncio.run(main())
