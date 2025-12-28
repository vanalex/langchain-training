from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pprint import pprint

load_dotenv()

def basic_chat():
    model = init_chat_model(model="gpt-5-nano")
    response = model.invoke("What is the capital of the moon?")
    print(response.content)
    pprint(response.response_metadata)


def basic_chat_with_temperature():
    model = init_chat_model(model="gpt-5-nano", temperature=1.0)
    response = model.invoke("What is the capital of the moon?")
    print(response.content)

def main():
    basic_chat()
    basic_chat_with_temperature()


if __name__ == "__main__":
    main()