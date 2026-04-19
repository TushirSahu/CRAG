import os
import nest_asyncio
nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

def build_vector_db():
    print("1. Loading Document...")
    
    file_path = "./new.pdf"
    
    # 💎 ENTERPRISE UPGRADE: Try LlamaParse for complex PDFs (tables, images, columns)
    llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if llama_key:
        print("🧠 LlamaParse API Key detected. Using Advanced VLM PDF extraction...")
        from llama_parse import LlamaParse
        
        parser = LlamaParse(
            api_key=llama_key,
            result_type="markdown", # Converts tables perfectly into Markdown!
            verbose=True
        )
        parsed_docs = parser.load_data(file_path)
        docs = [Document(page_content=doc.text, metadata={"source": file_path}) for doc in parsed_docs]
    else:
        print("⚠️ No LlamaCloud API Key found. Falling back to basic PyPDFLoader...")
        print("   (To fix messy table extraction, get a free key at cloud.llamaindex.ai)")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
    print(f"Loaded {len(docs)} highly-structured chunks.")

    print("\nChunking Document into semantic units...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Split the document into {len(chunks)} chunks.")

    print("\n2. Creating Embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("\n3. Storing in Chroma DB...")
    vector_db = Chroma()
    vector_db.from_documents(documents=chunks, embedding=embeddings, persist_directory="./data/chroma_db")
    print("✅ Chroma DB built successfully and saved to ./data/chroma_db")

if __name__ == "__main__":
    build_vector_db()
