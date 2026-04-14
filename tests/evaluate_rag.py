import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
    context_recall
)
# Ensure we have our environment variables (LangSmith, Google API, etc.)
from dotenv import load_dotenv
from src.agent.graph import app as crag_app

load_dotenv()

def run_evaluation():
    print("🚀 Starting Ragas Evaluation...")

    # 1. Define a small golden dataset of questions and their ground truth answers
    # In a real scenario, you'd load this from a CSV or JSON file.
    eval_data = [
        {
            "question": "What is the main topic discussed in the machine learning document?",
            "ground_truth": "The document primarily discusses loss functions, specifically focusing on error measurement in models."
        },
        {
            "question": "What is an ORDER Loss function?",
            "ground_truth": "There is no widely recognized standard loss function called 'ORDER Loss' in mainstream machine learning literature."
        }
    ]

    results_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    # 2. Generate predictions using our CRAG LangGraph pipeline
    print("⏳ Generating answers via CRAG Agent for evaluation...")
    for item in eval_data:
        question = item["question"]
        print(f"\nProcessing: {question}")
        
        inputs = {"question": question, "retry_count": 0, "chat_history": []}
        final_answer = ""
        final_contexts = []
        
        # Run the graph
        for output in crag_app.stream(inputs):
            for key, value in output.items():
                if key == "generate":
                    final_answer = value["generation"]
                    # Extract the raw text context from the documents
                    final_contexts = [doc["content"] for doc in value.get("documents", [])]

        results_data["question"].append(question)
        results_data["answer"].append(final_answer)
        results_data["contexts"].append(final_contexts if final_contexts else [""]) # Ragas expects a list of strings
        results_data["ground_truth"].append(item["ground_truth"])

    # 3. Convert to HuggingFace Dataset format (required by Ragas)
    dataset = Dataset.from_dict(results_data)
    
    # 4. Run the Evaluation
    print("\n📈 Running Ragas Metrics (Faithfulness, Relevancy, Precision, Recall)...")
    
    # Note: Ragas will use your default LangChain llm (OpenAI/Google depending on config) to grade the answers
    result = evaluate(
        dataset = dataset, 
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )
    
    # 5. Output the results
    print("\n✅ Evaluation Complete!")
    print(result)
    
    # Save to CSV for the interview portfolio 
    df = result.to_pandas()
    df.to_csv("ragas_evaluation_results.csv", index=False)
    print("💾 Detailed results saved to 'ragas_evaluation_results.csv'")

if __name__ == "__main__":
    run_evaluation()
