"""Module demonstrating a multi-agent wedding planning system.

This module shows how to:
- Build specialized subagents (travel, venue, playlist)
- Coordinate multiple agents through a main coordinator
- Integrate external services (MCP, Tavily, SQL databases)
- Manage shared state across agents
- Delegate complex tasks to specialized agents
"""

import asyncio
from pprint import pprint
from typing import Any, Dict

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_community.utilities import SQLDatabase
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import Command
from tavily import TavilyClient

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-5-nano"
MCP_SERVER_NAME = "travel_server"
MCP_SERVER_URL = "https://mcp.kiwi.com"
DB_PATH = "sqlite:///resources/Chinook.db"


class WeddingState(AgentState):
    """Custom state schema for wedding planning."""

    origin: str
    destination: str
    guest_count: str
    genre: str


async def initialize_mcp_client() -> MultiServerMCPClient:
    """Initialize MCP client for travel services.

    Returns:
        Configured MultiServerMCPClient instance
    """
    client = MultiServerMCPClient(
        {
            MCP_SERVER_NAME: {
                "transport": "streamable_http",
                "url": MCP_SERVER_URL,
            }
        }
    )
    return client


def initialize_tavily_client() -> TavilyClient:
    """Initialize Tavily client for web search.

    Returns:
        Configured TavilyClient instance
    """
    return TavilyClient()


def initialize_database() -> SQLDatabase:
    """Initialize SQL database connection.

    Returns:
        SQLDatabase instance connected to Chinook database
    """
    return SQLDatabase.from_uri(DB_PATH)


def create_search_tool(tavily_client: TavilyClient):
    """Create web search tool using Tavily.

    Args:
        tavily_client: Initialized Tavily client

    Returns:
        Web search tool
    """

    @tool
    def web_search(query: str) -> Dict[str, Any]:
        """Search the web for information.

        Args:
            query: Search query string

        Returns:
            Search results from Tavily
        """
        return tavily_client.search(query)

    return web_search


def create_database_tool(db: SQLDatabase):
    """Create database query tool.

    Args:
        db: Initialized SQLDatabase instance

    Returns:
        Database query tool
    """

    @tool
    def query_playlist_db(query: str) -> str:
        """Query the database for playlist information.

        Args:
            query: SQL query string

        Returns:
            Query results or error message
        """
        try:
            return db.run(query)
        except Exception as e:
            return f"Error querying database: {e}"

    return query_playlist_db


def create_travel_agent(tools: list) -> Any:
    """Create travel agent for flight searches.

    Args:
        tools: List of MCP tools for travel services

    Returns:
        Configured travel agent
    """
    system_prompt = """
    You are a travel agent. Search for flights to the desired destination wedding location.
    You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
    - Price (lowest, economy class)
    - Duration (shortest)
    - Date (time of year which you believe is best for a wedding at this location)
    To make things easy, only look for one ticket, one way.
    You may need to make multiple searches to iteratively find the best options.
    You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
    Once you have found the best options, let the user know your shortlist of options.
    """
    return create_agent(model=DEFAULT_MODEL, tools=tools, system_prompt=system_prompt)


def create_venue_agent(web_search_tool) -> Any:
    """Create venue agent for venue searches.

    Args:
        web_search_tool: Web search tool for finding venues

    Returns:
        Configured venue agent
    """
    system_prompt = """
    You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
    You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
    - Price (lowest)
    - Capacity (exact match)
    - Reviews (highest)
    You may need to make multiple searches to iteratively find the best options.
    """
    return create_agent(
        model=DEFAULT_MODEL, tools=[web_search_tool], system_prompt=system_prompt
    )


def create_playlist_agent(db_tool) -> Any:
    """Create playlist agent for music curation.

    Args:
        db_tool: Database query tool for accessing music database

    Returns:
        Configured playlist agent
    """
    system_prompt = """
    You are a playlist specialist. Query the sql database and curate the perfect playlist for a wedding given a genre.
    Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
    If you run into errors when querying the database, try to fix them by making changes to the query.
    Do not come back empty handed, keep trying to query the db until you find a list of songs.
    You may need to make multiple queries to iteratively find the best options.
    """
    return create_agent(model=DEFAULT_MODEL, tools=[db_tool], system_prompt=system_prompt)


def create_coordinator_tools(travel_agent, venue_agent, playlist_agent):
    """Create tools for the coordinator to delegate to subagents.

    Args:
        travel_agent: Travel agent instance
        venue_agent: Venue agent instance
        playlist_agent: Playlist agent instance

    Returns:
        List of coordinator tools
    """

    @tool
    async def search_flights(runtime: ToolRuntime) -> str:
        """Travel agent searches for flights to the desired destination wedding location.

        Args:
            runtime: Runtime context with state

        Returns:
            Flight search results
        """
        origin = runtime.state["origin"]
        destination = runtime.state["destination"]
        response = await travel_agent.ainvoke(
            {"messages": [HumanMessage(content=f"Find flights from {origin} to {destination}")]}
        )
        return response["messages"][-1].content

    @tool
    def search_venues(runtime: ToolRuntime) -> str:
        """Venue agent chooses the best venue for the given location and capacity.

        Args:
            runtime: Runtime context with state

        Returns:
            Venue search results
        """
        destination = runtime.state["destination"]
        capacity = runtime.state["guest_count"]
        query = f"Find wedding venues in {destination} for {capacity} guests"
        response = venue_agent.invoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    @tool
    def suggest_playlist(runtime: ToolRuntime) -> str:
        """Playlist agent curates the perfect playlist for the given genre.

        Args:
            runtime: Runtime context with state

        Returns:
            Playlist suggestions
        """
        genre = runtime.state["genre"]
        query = f"Find {genre} tracks for wedding playlist"
        response = playlist_agent.invoke({"messages": [HumanMessage(content=query)]})
        return response["messages"][-1].content

    @tool
    def update_state(
        origin: str,
        destination: str,
        guest_count: str,
        genre: str,
        runtime: ToolRuntime,
    ) -> Command:
        """Update the state when you know all of the values: origin, destination, guest_count, genre.

        Args:
            origin: Origin city
            destination: Destination city
            guest_count: Number of guests
            genre: Music genre preference
            runtime: Runtime context

        Returns:
            Command object with state updates
        """
        return Command(
            update={
                "origin": origin,
                "destination": destination,
                "guest_count": guest_count,
                "genre": genre,
                "messages": [
                    ToolMessage(
                        "Successfully updated state", tool_call_id=runtime.tool_call_id
                    )
                ],
            }
        )

    return [search_flights, search_venues, suggest_playlist, update_state]


def create_coordinator(coordinator_tools):
    """Create the main wedding coordinator agent.

    Args:
        coordinator_tools: Tools for delegating to subagents

    Returns:
        Configured coordinator agent
    """
    system_prompt = """
    You are a wedding coordinator. Delegate tasks to your specialists for flights, venues and playlists.
    First find all the information you need to update the state. Once that is done you can delegate the tasks.
    Once you have received their answers, coordinate the perfect wedding for me.
    """
    return create_agent(
        model=DEFAULT_MODEL,
        tools=coordinator_tools,
        state_schema=WeddingState,
        system_prompt=system_prompt,
    )


async def demo_wedding_planner():
    """Demonstrate the multi-agent wedding planning system."""
    print("\n=== Demo: Wedding Planner Multi-Agent System ===\n")

    # Initialize external services
    mcp_client = await initialize_mcp_client()
    tavily_client = initialize_tavily_client()
    database = initialize_database()

    # Get MCP tools for travel
    mcp_tools = await mcp_client.get_tools()

    # Create specialized tools
    web_search_tool = create_search_tool(tavily_client)
    db_query_tool = create_database_tool(database)

    # Create subagents
    travel_agent = create_travel_agent(mcp_tools)
    venue_agent = create_venue_agent(web_search_tool)
    playlist_agent = create_playlist_agent(db_query_tool)

    # Create coordinator
    coordinator_tools = create_coordinator_tools(
        travel_agent, venue_agent, playlist_agent
    )
    coordinator = create_coordinator(coordinator_tools)

    # Test the system
    user_request = (
        "I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre"
    )
    print(f"User Request: {user_request}\n")

    response = await coordinator.ainvoke(
        {"messages": [HumanMessage(content=user_request)]}
    )

    print("\n=== Full Response ===")
    pprint(response)

    print("\n=== Final Answer ===")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(demo_wedding_planner())
