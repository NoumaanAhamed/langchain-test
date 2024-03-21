from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import HumanMessage,AIMessage

# Create a simple chat with youtube application

url = "https://www.youtube.com/watch?v=uKLCzhvZLyA"
website_url = "https://python.langchain.com/docs/get_started/quickstart#diving-deeper"

# Step 1: Create a chat model
llm = ChatOpenAI()

# Step 2: Create a document loader
def youtube_loader(url):
    loader = YoutubeLoader.from_youtube_url(url)
    return loader.load()

def web_loader(url):
    loader = WebBaseLoader(url)
    return loader.load()


# Step 3: Create a document splitter
splitter = RecursiveCharacterTextSplitter()

# Step 4: Create a document embeddings
embeddings = OpenAIEmbeddings()

# Split the documents into smaller parts
# docs = youtube_loader(url)
docs = web_loader(website_url)
documents = splitter.split_documents(docs)

# Step 5: Create a vector store
vector = FAISS.from_documents(documents, embeddings)

# Step 6: Create a prompt template

retriever_prompt = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder(variable_name='chat_history'),
     ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")]
)

# Step 7: Create a Document Chain
# chain = create_stuff_documents_chain(llm,doc_prompt)

# Step 8: Create a retrieval chain
retriever = vector.as_retriever()
# fetch relevant documents
retriever_chain = create_history_aware_retriever(llm,retriever,retriever_prompt)

# Step 9: Create a chat chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(llm,prompt)

chat_chain = create_retrieval_chain(retriever_chain, document_chain)

chat_history = [HumanMessage(content="Can LangSmith help test my LLM applications?"), AIMessage(content="Yes!")]
# Step 10: Invoke the chain
response = chat_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me How?"})
print(response['answer'])

