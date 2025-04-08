from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os

load_dotenv()

if __name__ == "__main__":
    print("retrieval")
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-4o-mini")
    query = "What is embedings ?"
    chain = PromptTemplate.from_template(template=query) | llm
    vector_store = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEXNAME"), embedding=embeddings
    )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    conbine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(),
        combine_docs_chain=conbine_docs_chain    )
    
    res= retrieval_chain.invoke({"input": query})
    print(res)
