from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages(
    [("system","You are a professional chatbot designed to help people learn programming using {language}."),
     ("ai","Hey there! I'm a professional chatbot designed to help people learn programming using {language}. How can I help you today?"),
     ("user","{input}"),
     ],)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

def main():
    print("Welcome!, Enter the language you want to learn")
    language = input("Enter the language: ")
    print("type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            break
        response = chain.invoke({"language": language, "input": user_input})
        print("AI: " + response)

if __name__ == "__main__":
    main()