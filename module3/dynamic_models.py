from dotenv import load_dotenv

load_dotenv()

import logging
from typing import Callable
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
MESSAGE_THRESHOLD = 10
LARGE_MODEL_NAME = "claude-sonnet-4-5"
STANDARD_MODEL_NAME = "gpt-5-nano"

# Initialize models with error handling
try:
    large_model = init_chat_model(LARGE_MODEL_NAME)
    logger.info(f"Initialized large model: {LARGE_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize large model: {e}")
    raise

try:
    standard_model = init_chat_model(STANDARD_MODEL_NAME)
    logger.info(f"Initialized standard model: {STANDARD_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to initialize standard model: {e}")
    raise


def select_model_by_message_count(message_count: int, threshold: int = MESSAGE_THRESHOLD):
    """
    Select appropriate model based on conversation length.

    Args:
        message_count: Number of messages in the conversation
        threshold: Message count threshold for switching to large model

    Returns:
        Selected model instance
    """
    if message_count > threshold:
        logger.info(f"Selecting large model (messages: {message_count} > {threshold})")
        return large_model
    else:
        logger.info(f"Selecting standard model (messages: {message_count} <= {threshold})")
        return standard_model


@wrap_model_call
def state_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """
    Middleware that dynamically selects model based on conversation length.

    Switches to a larger model when the conversation exceeds a threshold,
    optimizing for context window and efficiency.

    Args:
        request: The model request containing state and messages
        handler: The next handler in the middleware chain

    Returns:
        ModelResponse from the selected model
    """
    try:
        message_count = len(request.messages)
        model = select_model_by_message_count(message_count)
        request = request.override(model=model)
        return handler(request)
    except Exception as e:
        logger.error(f"Error in state_based_model middleware: {e}")
        raise


def create_dynamic_agent(
    base_model: str = STANDARD_MODEL_NAME,
    system_prompt: str = "You are a helpful assistant."
):
    """
    Create an agent with dynamic model selection middleware.

    Args:
        base_model: The base model to use for agent initialization
        system_prompt: System prompt for the agent

    Returns:
        Configured agent with dynamic model selection
    """
    from langchain.agents import create_agent

    return create_agent(
        model=base_model,
        middleware=[state_based_model],
        system_prompt=system_prompt
    )


def run_example():
    """Demonstrate dynamic model selection with short and long conversations."""
    from langchain.messages import HumanMessage, AIMessage

    # Create agent with dynamic model selection
    agent = create_dynamic_agent(
        system_prompt="You are roleplaying a real life helpful office intern."
    )

    # Example 1: Short conversation (uses standard model)
    logger.info("=== Example 1: Short conversation ===")
    response = agent.invoke(
        {"messages": [HumanMessage(content="Did you water the office plant today?")]}
    )

    print(f"Response: {response['messages'][-1].content}")
    print(f"Model used: {response['messages'][-1].response_metadata['model_name']}\n")

    # Example 2: Long conversation (uses large model)
    logger.info("=== Example 2: Long conversation ===")
    response = agent.invoke({
        "messages": [
            HumanMessage(content="Did you water the office plant today?"),
            AIMessage(content="Yes, I gave it a light watering this morning."),
            HumanMessage(content="Has it grown much this week?"),
            AIMessage(content="It's sprouted two new leaves since Monday."),
            HumanMessage(content="Are the leaves still turning yellow on the edges?"),
            AIMessage(content="A little, but it's looking healthier overall."),
            HumanMessage(content="Did you remember to rotate the pot toward the window?"),
            AIMessage(content="I rotated it a quarter turn so it gets more even light."),
            HumanMessage(content="How often should we be fertilizing this plant?"),
            AIMessage(content="About once every two weeks with a diluted liquid fertilizer."),
            HumanMessage(content="When should we expect to have to replace the pot?")
        ]
    })

    print(f"Response: {response['messages'][-1].content}")
    print(f"Model used: {response['messages'][-1].response_metadata['model_name']}")


if __name__ == "__main__":
    run_example()