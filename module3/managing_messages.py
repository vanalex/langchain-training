from pprint import pprint
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import SummarizationMiddleware, before_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

load_dotenv()

def summarize_messages():
    agent = create_agent(
        model="gpt-5-nano",
        checkpointer=InMemorySaver(),
        middleware=[
            SummarizationMiddleware(
                model="gpt-4o-mini",
                trigger=("tokens", 100),
                keep=("messages", 1)
            )
        ],
    )

    response = agent.invoke(
        {"messages": [
            HumanMessage(content="What is the capital of the moon?"),
            AIMessage(content="The capital of the moon is Lunapolis."),
            HumanMessage(content="What is the weather in Lunapolis?"),
            AIMessage(content="Skies are clear, with a high of 120C and a low of -100C."),
            HumanMessage(content="How many cheese miners live in Lunapolis?"),
            AIMessage(content="There are 100,000 cheese miners living in Lunapolis."),
            HumanMessage(content="Do you think the cheese miners' union will strike?"),
            AIMessage(content="Yes, because they are unhappy with the new president."),
            HumanMessage(
                content="If you were Lunapolis' new president how would you respond to the cheese miners' union?"),
        ]},
        {"configurable": {"thread_id": "1"}}
    )

    pprint(response)



@before_agent
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Remove all the tool messages from the state"""
    messages = state["messages"]

    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    return {"messages": [RemoveMessage(id=m.id) for m in tool_messages]}


def trim_and_delete_messages():
    agent = create_agent(
        model="gpt-5-nano",
        checkpointer=InMemorySaver(),
        middleware=[trim_messages],
    )
    response = agent.invoke(
        {"messages": [
            HumanMessage(content="My device won't turn on. What should I do?"),
            ToolMessage(content="blorp-x7 initiating diagnostic ping…", tool_call_id="1"),
            AIMessage(content="Is the device plugged in and turned on?"),
            HumanMessage(content="Yes, it's plugged in and turned on."),
            ToolMessage(content="temp=42C voltage=2.9v … greeble complete.", tool_call_id="2"),
            AIMessage(content="Is the device showing any lights or indicators?"),
            HumanMessage(content="What's the temperature of the device?")
        ]},
        {"configurable": {"thread_id": "2"}}
    )

    pprint(response)

def main():
    #summarize_messages()
    trim_and_delete_messages()




if __name__ == "__main__":
    main()