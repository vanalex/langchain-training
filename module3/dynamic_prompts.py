from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()
from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent

@dataclass
class LanguageContext:
    user_language: str = "English"

@dynamic_prompt
def user_language_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_language = request.runtime.context.user_language
    base_prompt = "You are a helpful assistant."

    if user_language != "English":
        return f"{base_prompt} only respond in {user_language}."
    elif user_language == "English":
        return base_prompt

agent = create_agent(
    model="gpt-5-nano",
    context_schema=LanguageContext,
    middleware=[user_language_prompt]
)

response = agent.invoke(
    {"message": [HumanMessage(content="Hello, how are you?")]},
    context=LanguageContext(user_language="Irish")
)

print(response["messages"][-1].content)

response = agent.invoke(
    {"message": [HumanMessage(content="Hello, how are you?")]},
    context=LanguageContext(user_language="Spanish")
)

print(response["messages"][-1].content)

response = agent.invoke(
    {"message": [HumanMessage(content="Hello, how are you?")]},
    context=LanguageContext(user_language="French")
)

print(response["messages"][-1].content)