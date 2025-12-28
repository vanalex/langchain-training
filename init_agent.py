"""Demonstration of LangChain model and agent initialization."""

from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage

load_dotenv()


def basic_model_invocation() -> None:
    """Initialize and invoke a basic chat model."""
    model = init_chat_model(model="gpt-5-nano")
    response = model.invoke("What's the capital of the Moon?")

    print(response.content)
    print("\nResponse metadata:")
    pprint(response.response_metadata)


def customized_model() -> None:
    """Demonstrate model customization with parameters."""
    model = init_chat_model(
        model="gpt-5-nano",
        temperature=1.0,  # Higher temperature for more creative responses
    )

    response = model.invoke("What's the capital of the Moon?")
    print(response.content)


def basic_agent_invocation() -> None:
    """Initialize and invoke agents with different methods."""
    # Three ways to create an agent
    model = init_chat_model(model="gpt-5-nano")
    agent1 = create_agent(model=model)  # Pass model instance
    agent2 = create_agent(model="claude-sonnet-4-5")  # Pass model name with keyword
    agent3 = create_agent("gpt-5-nano")  # Pass model name directly

    # Use the third agent for demonstration
    response = agent3.invoke(
        {"messages": [HumanMessage(content="What's the capital of the Moon?")]}
    )

    print("Full response:")
    pprint(response)
    print(f"\nLast message content: {response['messages'][-1].content}")


def conversational_agent() -> None:
    """Demonstrate multi-turn conversation with an agent."""
    agent = create_agent("gpt-5-nano")

    response = agent.invoke(
        {
            "messages": [
                HumanMessage(content="What's the capital of the Moon?"),
                AIMessage(content="The capital of the Moon is Luna City."),
                HumanMessage(content="Interesting, tell me more about Luna City"),
            ]
        }
    )

    pprint(response)


def streaming_output() -> None:
    """Demonstrate streaming output from an agent."""
    agent = create_agent("gpt-5-nano")

    print("Streaming response:")
    for token, metadata in agent.stream(
        {"messages": [HumanMessage(content="Tell me all about Luna City, the capital of the Moon")]},
        stream_mode="messages",
    ):
        # token is a message chunk with token content
        # metadata contains which node produced the token
        if token.content:
            print(token.content, end="", flush=True)

    print()  # New line after streaming


def main() -> None:
    """Run all model and agent examples."""
    print("=== Basic Model Invocation ===")
    basic_model_invocation()

    print("\n=== Customized Model ===")
    customized_model()

    print("\n=== Basic Agent Invocation ===")
    basic_agent_invocation()

    print("\n=== Conversational Agent ===")
    conversational_agent()

    print("\n=== Streaming Output ===")
    streaming_output()


if __name__ == "__main__":
    main()