"""Module demonstrating runtime context in LangChain agents.

This module shows how to:
- Define custom context schemas using dataclasses
- Access runtime context from tools
- Pass context to agent invocations
- Use context for maintaining user preferences
"""

from dataclasses import dataclass
from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

# Load environment variables
load_dotenv()


@dataclass
class ColourContext:
    """Context schema for storing user colour preferences."""

    favourite_colour: str = "blue"
    least_favourite_colour: str = "yellow"


@tool
def get_favourite_colour(runtime: ToolRuntime) -> str:
    """Get the favourite colour of the user.

    Args:
        runtime: Tool runtime context containing user preferences

    Returns:
        The user's favourite colour
    """
    return runtime.context.favourite_colour


@tool
def get_least_favourite_colour(runtime: ToolRuntime) -> str:
    """Get the least favourite colour of the user.

    Args:
        runtime: Tool runtime context containing user preferences

    Returns:
        The user's least favourite colour
    """
    return runtime.context.least_favourite_colour


def demo_context_without_tools():
    """Demonstrate basic context usage without explicit tools."""
    print("\n=== Demo: Context Without Tools ===\n")

    agent = create_agent(model="gpt-5-nano", context_schema=ColourContext)

    response = agent.invoke(
        {"messages": [HumanMessage(content="What is my favourite colour?")]},
        context=ColourContext(),
    )

    pprint(response)


def demo_context_with_tools():
    """Demonstrate accessing context through tools."""
    print("\n=== Demo: Context With Tools ===\n")

    agent = create_agent(
        model="gpt-5-nano",
        tools=[get_favourite_colour, get_least_favourite_colour],
        context_schema=ColourContext,
    )

    # Use default context values
    print("Using default context:")
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is my favourite colour?")]},
        context=ColourContext(),
    )
    pprint(response)

    # Override context with custom values
    print("\nUsing custom context:")
    response = agent.invoke(
        {"messages": [HumanMessage(content="What is my favourite colour?")]},
        context=ColourContext(favourite_colour="green"),
    )
    pprint(response)


if __name__ == "__main__":
    demo_context_without_tools()
    demo_context_with_tools()
