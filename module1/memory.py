from pprint import pprint

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

def no_memory():
    agent = create_agent(
        model="gpt-5-nano"
    )

    question = HumanMessage(content="Hello, my name is Alex and my favorite color is yellow")
    response = agent.invoke({"messages": [question]})
    pprint(response["messages"])

    question = HumanMessage(content="What is your favorite color?")
    response = agent.invoke({"messages": [question]})
    pprint(response["messages"])

def memory():
    agent = create_agent(
        model="gpt-5-nano",
        checkpointer=InMemorySaver(),
    )

    question = HumanMessage(content="Hello, my name is Alex and my favorite color is yellow")
    config = {"configurable": {"thread_id": "1"}}
    response = agent.invoke({"messages": [question]}, config,)
    pprint(response["messages"])

    question = HumanMessage(content="What is your favorite color?")
    response = agent.invoke({"messages": [question]}, config)
    pprint(response["messages"])


if __name__ == '__main__':
    memory()