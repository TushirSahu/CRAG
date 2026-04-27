
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import networkx as nx
import pickle
from typing import List,TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
import os
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever

load_dotenv()

class GraphState(TypedDict):
    question: str
    chat_history: List[dict]
    generation: str
    web_search_needed: bool
    documents: List[dict]
    retry_count: int


llm = ChatGoogleGenerativeAI(model="gemma-3-4b-it", temperature=0)

finetuned_model_path = "./data/finetuned-domain-embeddings"
if os.path.exists(finetuned_model_path):
    print("🧠 Loading fine-tuned domain embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name=finetuned_model_path)
else:
    print("⚠️ Fine-tuned embeddings not found. Falling back to base HuggingFace model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma(persist_directory="./data/chroma_db", embedding_function=embeddings)

class CustomEnsembleRetriever(BaseRetriever):
    retrievers: list
    weights: list
    c: int = 60

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        from collections import defaultdict
        
        rrf_scores = defaultdict(float)
        doc_map = {}
        
        for weight, retriever in zip(self.weights, self.retrievers):
            docs = retriever.invoke(query)
            for rank, doc in enumerate(docs):
                clean_content = doc.page_content.strip()
                rrf_scores[clean_content] += weight * (1.0 / (rank + 1 + self.c))
                doc_map[clean_content] = doc
                
        # Sort by RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[content] for content, score in sorted_docs[:3]]


def get_hybrid_retriever():
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    try:
        db_data = vector_db.get()
        docs = [Document(page_content=c, metadata=m) for c, m in zip(db_data['documents'], db_data['metadatas'])]
        if docs:
            bm25 = BM25Retriever.from_documents(docs)
            bm25.k = 3
            return CustomEnsembleRetriever(retrievers=[bm25, vector_retriever], weights=[0.5, 0.5])
    except Exception as e:
        pass
    return vector_retriever

def extract_graph_context(question: str) -> str:
    graph_path = "./data/graph_db.pkl"
    if not os.path.exists(graph_path):
        return ""
    try:
        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)
        nodes = [str(n) for n in list(graph.nodes)[:15]]
        edges = [f"{u} -> {v}" for u, v in list(graph.edges)[:15]]
        return f"Graph Knowledge: Linked Entities: {nodes}. Relationships: {edges}"
    except Exception:
        return ""



def retrieve(state: GraphState):
    print("--Retrieving via Hybrid Search & GraphRAG--")
    question = state["question"]
    
    hybrid_retriever = get_hybrid_retriever()
    retrieved_docs = hybrid_retriever.invoke(question)
    documents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]
    
    graph_context = extract_graph_context(question)
    if graph_context:
        documents.append({"content": graph_context, "metadata": {"source": "Neo4j GraphRAG (Simulated)"}})

    return {"documents": documents, "question": question}


def grade_documents(state: GraphState):
    print("--Grading Documents--")
    question = state["question"]
    documents = state["documents"]

    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
    prompt = ChatPromptTemplate.from_template(
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "Question: {question}\n"
        "Document: {document}\n"
        "If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n"
        "Give a binary score 'yes' or 'no'."
    )
    retrieval_grader = prompt | llm | StrOutputParser()
    filtered_docs = []
    web_search_needed = False

    for doc in documents:
        score = retrieval_grader.invoke({"question": question, "document": doc["content"]})

        if "yes" in score.strip().lower():
            filtered_docs.append(doc)
            print(f"Document graded as relevant: {doc['content'][:50]}...")
        time.sleep(3)

    if not filtered_docs:
        print("No relevant documents found. Web search needed.")
        web_search_needed = True

    return {"documents": filtered_docs, "web_search_needed": web_search_needed, "question": question} 

def web_search(state: GraphState):
    print("--WEB SEARCH FALLBACK--")
    question = state["question"]
    if state.get("retry_count", 0) == 0:
        documents = state.get("documents", [])
    else:
        documents = []
    tool = TavilySearchResults(max_results=2)
    docs = tool.invoke({"query": question})
    
    if isinstance(docs, str):
        documents.append({"content": docs, "metadata": {"source": "Web Search"}})
        print("Web search results added as raw string.")
    elif isinstance(docs, list):
        for d in docs:
            if isinstance(d, dict) and "content" in d:
                source = d.get("url", "Web Search")
                documents.append({"content": d["content"], "metadata": {"source": source}})
            else:
                documents.append({"content": str(d), "metadata": {"source": "Web Search"}})
        print(f"Added {len(docs)} web search results.")
    else:
        documents.append({"content": str(docs), "metadata": {"source": "Web Search"}})
        print("Web search results added as raw format.")

    return {"documents": documents, "question": question}


def generate(state: GraphState):
    print("--Generating Answer--")
    question = state["question"]
    documents = state["documents"]
    chat_history = state.get("chat_history", [])

    history_text = ""
    if chat_history:
        history_text = "Previous Conversation:\n"
        for msg in chat_history[-4:]:  # keep last 4 messages
            history_text += f"{msg['role'].capitalize()}: {msg['content']}\n"

    context_blocks = []
    for idx, doc in enumerate(documents):
        source = doc['metadata'].get('source', 'Unknown')
        page = doc['metadata'].get('page', 'N/A')
        context_blocks.append(f"[Source {idx+1}] ({source}, Page {page}):\n{doc['content']}")
    
    context = "\n\n".join(context_blocks)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based strictly on the following context. "
        "Provide a highly detailed, comprehensive, and in-depth response. "
        "Synthesize all available information to give a thorough explanation. "
        "ALWAYS cite your sources inline using [Source 1], [Source 2], etc. "
        "At the very end of your answer, include a '### References' section that lists all the sources you referenced. "
        "When listing the references, simply write the name of the source EXACTLY as it is provided in the context blocks below (e.g., '[Source 1] - MyPDF.pdf (Page 6)'). Do NOT attempt to create hyperlinks, and do NOT add apologetic filler text like 'Since I cannot access external URLs'.\n\n"
        "{history_text}\n"
        "Context:\n"
        "{context}\n\n"
        "Question: {question}"
    )
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": context, "question": question, "history_text": history_text})
    return {"generation": generation, "question": question, "documents": documents}

def rewrite_query(state: GraphState):
    print("---REWRITING QUERY---")
    question = state["question"]
    
    class RewrittenQuery(BaseModel):
        optimized_query: str = Field(description="The optimized search engine query. No conversational text.")
        
    parser = JsonOutputParser(pydantic_object=RewrittenQuery)
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert web search optimizer. Your ONLY job is to look at the user's "
        "initial question and rewrite it into a highly effective search engine query. \n"
        "{format_instructions}\n"
        "Initial question: {question}"
    )
    
    rewriter = prompt | llm | parser 
    
    result = rewriter.invoke({
        "question": question,
        "format_instructions": parser.get_format_instructions()
    })
    
    clean_query = result["optimized_query"]
    current_retries = state.get("retry_count", 0)
    
    return {
        "question": clean_query, 
        "retry_count": current_retries + 1
    }

def decide_to_generate(state: GraphState):
    print("---DECIDING NEXT STEP---")
    if state.get("web_search_needed", False):
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT, ROUTE TO WEB SEARCH---")
        return "web_search"
    else:
        print("---DECISION: GENERATE---")
        return "generate"
    

def decider_after_web_search(state: GraphState):
    print("---DECIDING NEXT STEP---")
    if state.get("retry_count", 0) >= 2:
        print("Max retries reached. Proceeding to generation with current context.")
        return "generate"

    print("Checking if web search results are sufficient...")
    return "rewrite_query"

def grade_generation_v_documents_and_question(state: GraphState):
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    class GradeHallucinations(BaseModel):
        binary_score: str = Field(description="Answer 'yes' if the generation is supported by the documents, 'no' otherwise.")

    parser = JsonOutputParser(pydantic_object=GradeHallucinations)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n"
        "Here are the facts:\n"
        "-------\n"
        "{documents}\n"
        "-------\n"
        "Here is the LLM generation:\n"
        "{generation}\n"
        "-------\n"
        "Give a binary score 'yes' or 'no' indicating whether the answer is grounded in / supported by the facts.\n"
        "{format_instructions}"
    )

    hallucination_grader = prompt | llm | parser
    docs_str = "\n\n".join([doc["content"] for doc in documents])
    
    try:
        score = hallucination_grader.invoke({
            "documents": docs_str, 
            "generation": generation,
            "format_instructions": parser.get_format_instructions()
        })
        grade = score["binary_score"]
    except Exception as e:
        grade = "yes"  # Fallback gracefully
        
    if grade.lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        return "useful"
    else:
        print("---DECISION: GENERATION IS HALLUCINATED... REWRITING QUERY---")
        if state.get("retry_count", 0) >= 2:
             print("Max retries reached. Accepting generation as best attempt.")
             return "useful"
        return "not useful"


workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_query", rewrite_query)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    }
)
workflow.add_conditional_edges(
    "web_search",
    decider_after_web_search,
    {
        "rewrite_query": "rewrite_query",
        "generate": "generate",
    }
)

workflow.add_edge("rewrite_query", "web_search")

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "not useful": "rewrite_query",
    }
)

app = workflow.compile()


if __name__ == "__main__":
    print("\n\n=== TEST 1: LOCAL KNOWLEDGE ===")
    inputs = {"question": "What is the main topic discussed in the provided document?"} 
    
    for output in app.stream(inputs):
        for key, value in output.items():
            pass # We let the nodes print their own status updates
            
    print("\n✅ Final Answer:")
    print(value["generation"])


    print("\n\n=== TEST 2: FORCING THE WEB FALLBACK ===")
    inputs = {"question": "What is the weather in Tokyo today?"}
    
    for output in app.stream(inputs):
        for key, value in output.items():
            pass
            
    print("\n✅ Final Answer:")
    print(value["generation"])