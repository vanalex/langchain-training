"""Module demonstrating multi-agent architecture in LangChain.

This module shows how to:
- Create specialized subagents with specific tools
- Wrap subagents as callable tools
- Build a main agent that orchestrates subagents
- Handle delegation and coordination between agents
"""

from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-5-nano"


@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number.

    Args:
        x: The number to calculate square root for

    Returns:
        The square root of x
    """
    return x**0.5


@tool
def square(x: float) -> float:
    """Calculate the square of a number.

    Args:
        x: The number to square

    Returns:
        The square of x
    """
    return x**2


def create_subagents():
    """Create specialized subagents with specific mathematical capabilities.

    Returns:
        Tuple of (subagent_1, subagent_2) where:
            - subagent_1 handles square root calculations
            - subagent_2 handles square calculations
    """
    subagent_1 = create_agent(model=DEFAULT_MODEL, tools=[square_root])
    subagent_2 = create_agent(model=DEFAULT_MODEL, tools=[square])

    return subagent_1, subagent_2


def create_subagent_tools(subagent_1, subagent_2):
    """Create tools that wrap subagent invocations.

    Args:
        subagent_1: Agent for square root calculations
        subagent_2: Agent for square calculations

    Returns:
        List of tools for calling subagents
    """

    @tool
    def call_subagent_1(x: float) -> float:
        """Call subagent 1 to calculate the square root of a number.

        Args:
            x: The number to calculate square root for

        Returns:
            The square root result as a string
        """
        response = subagent_1.invoke(
            {"messages": [HumanMessage(content=f"Calculate the square root of {x}")]}
        )
        return response["messages"][-1].content

    @tool
    def call_subagent_2(x: float) -> float:
        """Call subagent 2 to calculate the square of a number.

        Args:
            x: The number to square

        Returns:
            The square result as a string
        """
        response = subagent_2.invoke(
            {"messages": [HumanMessage(content=f"Calculate the square of {x}")]}
        )
        return response["messages"][-1].content

    return [call_subagent_1, call_subagent_2]


def create_main_agent(subagent_tools):
    """Create the main orchestrator agent.

    Args:
        subagent_tools: List of tools for delegating to subagents

    Returns:
        Main agent that can coordinate subagent calls
    """
    system_prompt = (
        "You are a helpful assistant who can call subagents to calculate "
        "the square root or square of a number."
    )

    return create_agent(
        model=DEFAULT_MODEL, tools=subagent_tools, system_prompt=system_prompt
    )


def demo_multiagent():
    """Demonstrate multi-agent coordination for mathematical operations."""
    print("\n=== Demo: Multi-Agent System ===\n")

    # Create subagents
    subagent_1, subagent_2 = create_subagents()

    # Create tools that wrap subagents
    subagent_tools = create_subagent_tools(subagent_1, subagent_2)

    # Create main orchestrator agent
    main_agent = create_main_agent(subagent_tools)

    # Test the multi-agent system
    question = "What is the square root of 456?"
    print(f"Question: {question}\n")

    response = main_agent.invoke({"messages": [HumanMessage(content=question)]})
    pprint(response)


if __name__ == "__main__":
    demo_multiagent()
