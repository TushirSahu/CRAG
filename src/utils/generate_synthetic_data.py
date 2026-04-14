import os
import pandas as pd
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Define the structured output we want from Gemini
class QA_Pair(BaseModel):
    question: str = Field(description="A highly technical question that can be answered by the context.")

class QA_Pairs(BaseModel):
    pairs: list[QA_Pair]

def generate_dataset():
    print("🚀 Starting synthetic data generation via Gemini...")
    
    # 1. Initialize LLM (Using Gemini 2.5 Flash for speed/cost efficiency)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    parser = JsonOutputParser(pydantic_object=QA_Pairs)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert researcher. Read the following text chunk and generate 2 highly specific, "
        "technical questions that can be answered strictly using this text.\n"
        "Format instructions:\n{format_instructions}\n\n"
        "Text chunk:\n{context}"
    )
    chain = prompt | llm | parser

    # 2. Get chunks from your local ChromaDB
    print("📚 Loading existing chunks from ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./data/chroma_db", embedding_function=embeddings)
    
    db_data = vector_db.get() 
    chunks = db_data.get('documents', [])
    
    if not chunks:
        print("⚠️ No documents found in ChromaDB! Please upload a PDF in the Streamlit app first.")
        return

    print(f"✅ Found total {len(chunks)} chunks in database. Generating questions for the first 15 for demonstration...")
    
    training_data = []
    
    # 3. Generate Questions (Limiting to 15 chunks to avoid API rate limits during testing)
    for i, chunk in enumerate(chunks[:15]):
        try:
            print(f"Processing chunk {i+1}...")
            result = chain.invoke({
                "context": chunk,
                "format_instructions": parser.get_format_instructions()
            })
            
            # Map the generated questions BACK to the original database chunk
            for item in result["pairs"]:
                training_data.append({
                    "question": item["question"],
                    "context": chunk
                })
            time.sleep(2) # Sleep shortly to avoid hitting Google API Free-Tier rate limits
        except Exception as e:
            print(f"Error on chunk {i+1}: {e}")

    # 4. Save to CSV
    if training_data:
        os.makedirs("./data", exist_ok=True)
        df = pd.DataFrame(training_data)
        output_csv = "./data/training_pairs.csv"
        df.to_csv(output_csv, index=False)
        print(f"\n🎉 Successfully generated {len(training_data)} training pairs!")
        print(f"💾 Saved to '{output_csv}'. You can now run finetune_embeddings.py!")
    else:
        print("❌ Failed to generate any training pairs.")

if __name__ == "__main__":
    generate_dataset()