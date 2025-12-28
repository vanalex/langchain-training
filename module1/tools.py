from typing import Any

from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pprint import pprint

load_dotenv()


@tool("square_root", description="Calculate the square root of a number")
def tool1(x: float) -> float:
    return x ** 0.5


def init_agent() -> Any:
    return create_agent(
        model="gpt-5-nano",
        tools=[tool1],
        system_prompt = "You are an arithmetic wizard. Use your tools to calculate the square root and square of any number."
    )

def main():
    question = HumanMessage(content="What is the square root of 467?")
    response = init_agent().invoke(
        {"messages": [question]}
    )

    print(response['messages'][-1].content)

    pprint(response['messages'])
    print(response["messages"][1].tool_calls)


if __name__ == '__main__':
    main()


