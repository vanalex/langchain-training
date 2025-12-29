import asyncio

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.messages import HumanMessage
from pprint import pprint

load_dotenv()

client = MultiServerMCPClient(
    {
        "travel_server": {
                "transport": "streamable_http",
                "url": "https://mcp.kiwi.com"
            }
    }
)
async def travel_agency():
    tools = await client.get_tools()

    agent = create_agent(
        "gpt-5-nano",
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt="You are a travel agent. No follow up questions."
    )

    config = {"configurable": {"thread_id": "1"}}

    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Get me a direct flight from San Francisco to Tokyo on March 31st")]},
        config
        )

    pprint(response)
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(travel_agency())