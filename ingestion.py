import os
from google import genai
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import getpass
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings.openai import OpenAIEmbeddings


load_dotenv()


if __name__ == "__main__":
    print("ingestion")
    loader = TextLoader(
        "/Users/sukumarnagaboosanam/AI/GitHub/intro-to-vector-dbs/mediumblog.txt"
    )
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(f"Loaded {len(docs)} documents")
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(
        docs, embeddings, index_name=os.getenv("PINECONE_INDEXNAME")
    )
