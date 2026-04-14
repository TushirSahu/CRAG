"""
Domain Adaptation for Embedding Models (Contrastive Learning)

This script demonstrates how to fine-tune the base HuggingFace embedding model 
('all-MiniLM-L6-v2') on your specific domain data.

Why do this?
Base embedding models are trained on Wikipedia and general knowledge. They struggle with 
highly technical vocabulary. By fine-tuning using Multiple Negatives Ranking Loss (MNRL), 
we force the model to map domain-specific questions closer to their highly technical 
paragraphs in the vector space, massively boosting ChromaDB's retrieval precision.
"""

import os
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def finetune_embeddings():
    print("🚀 Loading base model for domain adaptation...")
    # 1. Load the base pre-trained model
    model_name = "all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)

    # 2. Prepare your Domain-Specific Training Data
    dataset_path = "./data/training_pairs.csv"
    train_examples = []

    print("📊 Loading domain-specific contrastive dataset...")
    
    if os.path.exists(dataset_path):
        # Load from the actual CSV you generated using Gemini
        df = pd.read_csv(dataset_path)
        for _, row in df.iterrows():
            # CSV must have 'question' and 'context' columns
            train_examples.append(InputExample(texts=[row['question'], row['context']]))
        print(f"✅ Successfully loaded {len(train_examples)} pairs from {dataset_path}")
    else:
        print(f"⚠️ Warning: '{dataset_path}' not found! Using dummy fallback data for demonstration.")
        train_examples = [
            InputExample(texts=[
                "How does the CRAG system handle hallucinations?", 
                "The Corrective RAG (CRAG) system implements a self-correction mechanism where an LLM grader evaluates the relevance of retrieved documents. If the relevance is low, it triggers a web search fallback."
            ]),
            InputExample(texts=[
                "What is the purpose of semantic caching in this architecture?", 
                "Semantic caching stores previous user queries and their generated answers in a local ChromaDB instance. When a semantically similar question is asked, it returns the cached answer instantly, bypassing the LLM to save API costs and reduce latency."
            ]),
            InputExample(texts=[
                "Why do we rewrite queries during the web search phase?", 
                "Query rewriting transforms a conversational user input into an optimized keyword search string. This ensures that the external search API, such as Tavily, retrieves the most accurate and dense information possible when local context fails."
            ])
        ]

    # 3. Create a DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)

    # 4. Define the Loss Function (Multiple Negatives Ranking Loss)
    # This is the gold standard for Retrieval/Search fine-tuning.
    # It pulls the question and correct passage together, while pushing ALL other passages in the batch away.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 5. Fine-Tune the Model
    output_path = "./data/finetuned-domain-embeddings"
    print(f"🧠 Fine-tuning model using Multiple Negatives Ranking Loss...")
    
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=20,
        warmup_steps=10,
        show_progress_bar=True,
        output_path=output_path
    )

    print(f"✅ Fine-tuning complete! Model saved to {output_path}")
    print("To use this in CRAG, update graph.py to load this local model path in HuggingFaceEmbeddings.")

if __name__ == "__main__":
    finetune_embeddings()
