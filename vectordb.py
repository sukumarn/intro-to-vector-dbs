from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma

if __name__ == "__main__":
    # Load the PDF document
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the PDF and create chunks
    loader = PyPDFLoader("handbook.pdf")
    text_splitter = CharacterTextSplitter(
        separator=".",
        chunk_size=250,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
        keep_separator=False,
    )

    pages = loader.load_and_split(text_splitter)
