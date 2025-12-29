"""Module demonstrating custom state management in LangChain agents.

This module shows how to:
- Define custom state schemas
- Update state using tools with Command objects
- Read state from runtime context
- Maintain conversation state across invocations
"""

from dotenv import load_dotenv
from pprint import pprint

from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

# Load environment variables
load_dotenv()


class CustomState(AgentState):
    """Custom agent state with favourite colour tracking."""

    favourite_colour: str


@tool
def update_favourite_colour(favourite_colour: str, runtime: ToolRuntime) -> Command:
    """Update the favourite colour of the user in the state once they've revealed it.

    Args:
        favourite_colour: The user's favourite colour to store
        runtime: Tool runtime context for accessing tool_call_id

    Returns:
        Command object with state updates
    """
    return Command(
        update={
            "favourite_colour": favourite_colour,
            "messages": [
                ToolMessage(
                    "Successfully updated favourite colour",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def read_favourite_colour(runtime: ToolRuntime) -> str:
    """Read the favourite colour of the user from the state.

    Args:
        runtime: Tool runtime context for accessing state

    Returns:
        The user's favourite colour or error message if not found
    """
    try:
        return runtime.state["favourite_colour"]
    except KeyError:
        return "No favourite colour found in state"


def demo_update_state():
    """Demonstrate updating state through tool execution."""
    print("\n=== Demo: Update State ===\n")

    agent = create_agent(
        "gpt-5-nano",
        tools=[update_favourite_colour],
        checkpointer=InMemorySaver(),
        state_schema=CustomState,
    )

    response = agent.invoke(
        {"messages": [HumanMessage(content="My favourite colour is green")]},
        {"configurable": {"thread_id": "1"}},
    )
    pprint(response)

    # Demonstrate initializing state directly
    response = agent.invoke(
        {
            "messages": [HumanMessage(content="Hello, how are you?")],
            "favourite_colour": "green",
        },
        {"configurable": {"thread_id": "10"}},
    )
    pprint(response)


def demo_read_state():
    """Demonstrate reading state from runtime context."""
    print("\n=== Demo: Read State ===\n")

    agent = create_agent(
        "gpt-5-nano",
        tools=[update_favourite_colour, read_favourite_colour],
        checkpointer=InMemorySaver(),
        state_schema=CustomState,
    )

    # Set the favourite colour
    response = agent.invoke(
        {"messages": [HumanMessage(content="My favourite colour is green")]},
        {"configurable": {"thread_id": "1"}},
    )
    pprint(response)

    # Read the favourite colour from state
    response = agent.invoke(
        {"messages": [HumanMessage(content="What's my favourite colour?")]},
        {"configurable": {"thread_id": "1"}},
    )
    pprint(response)


if __name__ == "__main__":
    demo_update_state()
    demo_read_state()
