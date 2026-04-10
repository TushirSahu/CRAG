from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class LocalSemanticCache():
    def __init__(self, threshold: float = 0.7):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.cache_db = Chroma(collection_name="semantic_cache",persist_directory="./data/cache_db", embedding_function=self.embeddings)
        self.threshold = threshold

    def check_cache(self, query: str):
        print("--Checking Cache--")
        results = self.cache_db.similarity_search_with_score(query, k=1)
        
        if results:
            doc,score = results[0]
            if score < self.threshold:
                print(f"Cache hit with similarity score: {score:.2f}")
                return doc.metadata["answer"]

        
        print("Cache miss")
        return None

    def add_to_cache(self, query: str, answer: str):
        print("--Adding to Cache--")
        self.cache_db.add_texts(texts=[query], metadatas=[{"answer": answer}])
        print("Added to cache successfully")