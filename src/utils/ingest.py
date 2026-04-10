import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_vector_db():
    print("1. Loading Document...")
    # Point this to your actual PDF file in your workspace
    loader = PyPDFLoader("./new.pdf")
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    print("/n Chunking Document...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap  = 50)
    chunks = text_splitter.split_documents(docs)
    print(f"Split the document into {len(chunks)} chunks.")

    print("2. Creating Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # chunk_embeddings = embeddings.embed_documents(chunks)

    print("3. Storing in Chroma DB...")
    vector_db = Chroma()
    vector_db.from_documents(documents=chunks, embedding=embeddings,persist_directory="./chroma_db")
    print("Chroma DB built successfully and saved to ./chroma_db")

if __name__ == "__main__":
    build_vector_db()
