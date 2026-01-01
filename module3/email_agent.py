from dotenv import load_dotenv
from dataclasses import dataclass
from langchain.agents import AgentState, create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage, HumanMessage
from langchain.agents.middleware import wrap_model_call, dynamic_prompt, HumanInTheLoopMiddleware
from langchain.agents.middleware import ModelRequest, ModelResponse
from typing import Callable
import os
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmailContext:
    """Context for email authentication credentials."""
    email_address: str = os.getenv("EMAIL_ADDRESS", "julie@example.com")
    password: str = os.getenv("EMAIL_PASSWORD", "passwd")


class AuthenticatedState(AgentState):
    """Agent state that tracks authentication status."""
    authenticated: bool


@tool
def check_inbox() -> str:
    """Check the inbox for recent emails."""
    logger.info("Checking inbox")
    return """
    Hi Julie,
    I'm going to be in town next week and was wondering if we could grab a coffee?
    - best, Jane (jane@example.com)
    """


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send a response email to the specified recipient."""
    logger.info(f"Sending email to {to}")
    return f"Email sent to {to} with subject '{subject}'"


@tool
def authenticate(email: str, password: str, runtime: ToolRuntime) -> Command:
    """
    Authenticate the user with the given email and password.

    Args:
        email: User's email address
        password: User's password
        runtime: Tool runtime context

    Returns:
        Command with updated authentication state
    """
    try:
        is_authenticated = (
            email == runtime.context.email_address and
            password == runtime.context.password
        )

        if is_authenticated:
            logger.info(f"Authentication successful for {email}")
            return Command(
                update={
                    "authenticated": True,
                    "messages": [
                        ToolMessage(
                            "Successfully authenticated",
                            tool_call_id=runtime.tool_call_id
                        )
                    ],
                }
            )
        else:
            logger.warning(f"Authentication failed for {email}")
            return Command(
                update={
                    "authenticated": False,
                    "messages": [
                        ToolMessage(
                            "Authentication failed - invalid credentials",
                            tool_call_id=runtime.tool_call_id
                        )
                    ],
                }
            )
    except Exception as e:
        logger.error(f"Error during authentication: {e}")
        return Command(
            update={
                "authenticated": False,
                "messages": [
                    ToolMessage(
                        f"Authentication error: {str(e)}",
                        tool_call_id=runtime.tool_call_id
                    )
                ],
            }
        )


@wrap_model_call
def dynamic_tool_call(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    Dynamically provide tools based on authentication status.

    Unauthenticated users only have access to the authenticate tool.
    Authenticated users have access to check_inbox and send_email tools.

    Args:
        request: The model request containing state
        handler: The next handler in the middleware chain

    Returns:
        ModelResponse from the handler with appropriate tools
    """
    try:
        authenticated = request.state.get("authenticated", False)

        if authenticated:
            tools = [check_inbox, send_email]
            logger.info("User authenticated - providing inbox and email tools")
        else:
            tools = [authenticate]
            logger.info("User not authenticated - providing only authenticate tool")

        request = request.override(tools=tools)
        return handler(request)
    except Exception as e:
        logger.error(f"Error in dynamic_tool_call: {e}")
        raise


AUTHENTICATED_PROMPT = "You are a helpful assistant that can check the inbox and send emails."
UNAUTHENTICATED_PROMPT = "You are a helpful assistant that can authenticate users. Ask the user for their email and password to authenticate."


@dynamic_prompt
def dynamic_prompt_func(request: ModelRequest) -> str:
    """
    Generate system prompt based on authentication status.

    Args:
        request: The model request containing state

    Returns:
        Appropriate system prompt based on authentication status
    """
    authenticated = request.state.get("authenticated", False)

    if authenticated:
        logger.debug("Using authenticated prompt")
        return AUTHENTICATED_PROMPT
    else:
        logger.debug("Using unauthenticated prompt")
        return UNAUTHENTICATED_PROMPT


def create_email_agent(
    model: str = "gpt-5-nano",
    require_approval: bool = True
):
    """
    Create an email agent with authentication and dynamic tool access.

    Args:
        model: The model to use for the agent
        require_approval: Whether to require human approval for tool calls

    Returns:
        Configured email agent
    """
    return create_agent(
        model,
        tools=[authenticate, check_inbox, send_email],
        state_schema=AuthenticatedState,
        context_schema=EmailContext,
        middleware=[
            dynamic_tool_call,
            dynamic_prompt_func,
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "authenticate": require_approval,
                    "check_inbox": require_approval,
                    "send_email": require_approval,
                }
            ),
        ],
    )


def run_example():
    """Demonstrate the email agent workflow."""
    from langgraph.checkpoint.memory import InMemorySaver

    # Create custom context
    context = EmailContext(
        email_address="user@example.com",
        password="secure_password"
    )

    agent = create_email_agent(
        require_approval=False  # Disable approval for demo
    )

    config = {
        "configurable": {
            "thread_id": "email_demo",
            "context": context
        }
    }

    # Example 1: Attempt to check inbox without authentication
    logger.info("=== Example 1: Unauthenticated request ===")
    response = agent.invoke(
        {"messages": [HumanMessage(content="Check my inbox")]},
        config=config
    )
    print(f"Response: {response['messages'][-1].content}\n")

    # Example 2: Authenticate
    logger.info("=== Example 2: Authenticate ===")
    response = agent.invoke(
        {"messages": [HumanMessage(content="Authenticate with user@example.com and password secure_password")]},
        config=config
    )
    print(f"Response: {response['messages'][-1].content}\n")

    # Example 3: Check inbox after authentication
    logger.info("=== Example 3: Check inbox after authentication ===")
    response = agent.invoke(
        {"messages": [HumanMessage(content="Check my inbox now")]},
        config=config
    )
    print(f"Response: {response['messages'][-1].content}\n")


if __name__ == "__main__":
    run_example()