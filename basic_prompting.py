"""Demonstration of LangChain prompting techniques."""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pydantic import BaseModel

load_dotenv()


def basic_prompting_example() -> None:
    """Demonstrate basic prompting without a system prompt."""
    agent = create_agent(model="gpt-5-nano")
    question = HumanMessage(content="What's the capital of the moon?")
    response = agent.invoke({"messages": [question]})
    print(response["messages"][1].content)


def system_prompt_example() -> None:
    """Demonstrate prompting with a system prompt."""
    system_prompt = "You are a science fiction writer, create a capital city at the users request."
    scifi_agent = create_agent(model="gpt-5-nano", system_prompt=system_prompt)

    question = HumanMessage(content="What's the capital of the moon?")
    response = scifi_agent.invoke({"messages": [question]})
    print(response["messages"][1].content)


def few_shot_prompting_example() -> None:
    """Demonstrate few-shot prompting with examples."""
    system_prompt = """
You are a science fiction writer, create a space capital city at the users request.

User: What is the capital of mars?
Scifi Writer: Marsialis

User: What is the capital of Venus?
Scifi Writer: Venusovia
"""
    scifi_agent = create_agent(model="gpt-5-nano", system_prompt=system_prompt)

    question = HumanMessage(content="What's the capital of the moon?")
    response = scifi_agent.invoke({"messages": [question]})
    print(response["messages"][1].content)


def structured_prompt_example() -> None:
    """Demonstrate structured prompts with explicit format instructions."""
    system_prompt = """
You are a science fiction writer, create a space capital city at the users request.

Please keep to the below structure.

Name: The name of the capital city

Location: Where it is based

Vibe: 2-3 words to describe its vibe

Economy: Main industries
"""
    scifi_agent = create_agent(model="gpt-5-nano", system_prompt=system_prompt)

    question = HumanMessage(content="What's the capital of the moon?")
    response = scifi_agent.invoke({"messages": [question]})
    print(response["messages"][1].content)


class CapitalInfo(BaseModel):
    """Pydantic model for structured capital city information."""

    name: str
    location: str
    vibe: str
    economy: str


def structured_output_example() -> None:
    """Demonstrate structured output using Pydantic models."""
    agent = create_agent(
        model="gpt-5-nano",
        system_prompt="You are a science fiction writer, create a capital city at the users request.",
        response_format=CapitalInfo,
    )

    question = HumanMessage(content="What is the capital of The Moon?")
    response = agent.invoke({"messages": [question]})

    capital_info = response["structured_response"]
    capital_name = capital_info.name
    capital_location = capital_info.location

    print(f"{capital_name} is a city located at {capital_location}")


def main() -> None:
    """Run all prompting examples."""
    print("=== Basic Prompting ===")
    basic_prompting_example()

    print("\n=== System Prompt ===")
    system_prompt_example()

    print("\n=== Few-Shot Prompting ===")
    few_shot_prompting_example()

    print("\n=== Structured Prompt ===")
    structured_prompt_example()

    print("\n=== Structured Output ===")
    structured_output_example()


if __name__ == "__main__":
    main()

