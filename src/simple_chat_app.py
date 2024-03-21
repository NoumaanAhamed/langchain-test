from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Chat with Youtube Videos

llm = ChatOpenAI()
splitter = RecursiveCharacterTextSplitter()
embeddings = OpenAIEmbeddings()

def main():
    print("Welcome to the simple chat application")
    url = input("Enter youtube url: ")
    loader = YoutubeLoader.from_youtube_url(url)
    docs = loader.load()
    documents = splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name='chat_history'),
     ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")])
    retriever = vector.as_retriever()
    retriever_chain = create_history_aware_retriever(llm,retriever,retriever_prompt)
    chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),])
    document_chain = create_stuff_documents_chain(llm,chat_prompt)
    chat_chain = create_retrieval_chain(retriever_chain,document_chain)
    while True:
        user_input = input("You: ")
        response = chat_chain.invoke({
            "input": user_input,
            "chat_history": []
        
        })
        print("Bot: ", response['answer'])
        if user_input == "exit":
            break

if __name__ == "__main__":
    main()



