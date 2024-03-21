from langchain_community.document_loaders.youtube import YoutubeLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain

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
print(docs)
documents = splitter.split_documents(docs)

# Step 5: Create a vector store
vector = FAISS.from_documents(documents, embeddings)

# Step 6: Create a prompt template

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Step 7: Create a Document Chain

chain = create_stuff_documents_chain(llm,prompt)

# Step 8: Create a retrieval chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, chain)

response = retrieval_chain.invoke({"input": "What is the video about?"})
print(response["answer"])

