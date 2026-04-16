import asyncio
import os
import sys

import nest_asyncio
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

nest_asyncio.apply()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.outputs import ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from src.agent.graph import app as crag_app
from src.agent.graph import embeddings as domain_embeddings

load_dotenv()

class GemmaRagasWrapper(ChatGoogleGenerativeAI):
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        n = kwargs.pop("n", getattr(self, "n", 1))
        kwargs.pop("candidate_count", None)
        
        original_n = getattr(self, "n", None)
        self.n = None
        
        try:
            if n == 1 or n is None:
                return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
            
            tasks = [
                super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                for _ in range(n)
            ]
            results = await asyncio.gather(*tasks)
            
            combined_generations = [gen for res in results for gen in res.generations]
            return ChatResult(generations=combined_generations, llm_output=results[0].llm_output)
        finally:
            self.n = original_n

EVALUATION_DATA = [
    {
        "question": "How does the CRAG system handle hallucinations?",
        "ground_truth": "The CRAG system uses a self-correction mechanism with an LLM grader that evaluates both document relevance and answer groundedness. If an answer is hallucinated or ungrounded, it triggers a query rewrite and web search fallback."
    },
    {
        "question": "What is the purpose of semantic caching in this architecture?",
        "ground_truth": "Semantic caching stores previous user queries and their generated answers locally in ChromaDB. It instantly returns cached answers for semantically similar questions, reducing API costs and latency to zero."
    }
]

def run_evaluation():
    print("🚀 Starting Ragas Evaluation...")

    results_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print("⏳ Generating answers via CRAG Agent for evaluation...")
    
    for item in EVALUATION_DATA:
        question = item["question"]
        print(f"\nProcessing: {question}")
        
        inputs = {"question": question, "retry_count": 0, "chat_history": []}
        final_answer = ""
        final_contexts = []
        
        for output in crag_app.stream(inputs):
            for key, value in output.items():
                if key == "generate":
                    final_answer = value["generation"]
                    final_contexts = [doc["content"] for doc in value.get("documents", [])]

        results_data["question"].append(question)
        results_data["answer"].append(final_answer)
        results_data["contexts"].append(final_contexts if final_contexts else [""])
        results_data["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(results_data)
    
    print("\n📈 Running Ragas Metrics (Faithfulness, Relevancy, Precision, Recall)...")

    gemini_llm = GemmaRagasWrapper(model="gemma-3-4b-it", temperature=0)
    
    result = evaluate(
        dataset=dataset, 
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=gemini_llm,
        embeddings=domain_embeddings,
        raise_exceptions=True,
        run_config=RunConfig(timeout=1800, max_workers=1)
    )
    
    print("\n✅ Evaluation Complete!")
    print(result)
    
    df = result.to_pandas()
    df.to_csv("ragas_evaluation_results.csv", index=False)
    print("💾 Detailed results saved to 'ragas_evaluation_results.csv'")

if __name__ == "__main__":
    run_evaluation()
