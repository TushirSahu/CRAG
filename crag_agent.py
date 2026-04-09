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

load_dotenv()

class GraphState(TypedDict):
    question: str
    chat_history: List[dict]
    generation: str
    web_search_needed: bool
    documents: List[dict]
    retry_count: int


llm = ChatGoogleGenerativeAI(model="gemma-3-1b-it", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})


#---NODE 1: Retrieve    
def retrieve(state: GraphState):
    print("--Retrieving from vector DB--")
    question = state["question"]
    retrieved_docs = retriever.invoke(question)
    documents = [{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]

    return {"documents": documents, "question": question}


#---NODE 2: Grade Documents
def grade_documents(state: GraphState):
    print("--Grading Documents--")
    question = state["question"]
    documents = state["documents"]

    class Grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")
    
    # structured_llm = llm.with_structured_output(Grade)
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

        if score.strip().lower() == "yes":
            filtered_docs.append(doc)
            print(f"Document graded as relevant: {doc['content'][:50]}...")
        time.sleep(3)

    if not filtered_docs:
        print("No relevant documents found. Web search needed.")
        web_search_needed = True

    return {"documents": filtered_docs, "web_search_needed": web_search_needed, "question": question} 

#---NODE 3: Web Search
def web_search(state: GraphState):
    print("--WEB SEARCH FALLBACK--")
    question = state["question"]
    if state.get("retry_count", 0) == 0:
        documents = state.get("documents", [])
    else:
        documents = []
    # documents = state["documents"]
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


#---NODE 4: Generate Answer
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

    # Combine docs with source citations
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
        "At the very end of your answer, include a '### References' section that lists all the sources you referenced.\n\n"
        "{history_text}\n"
        "Context:\n"
        "{context}\n\n"
        "Question: {question}"
    )
    rag_chain = prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"context": context, "question": question, "history_text": history_text})
    return {"generation": generation, "question": question, "documents": documents}

#---Node 5: rewriting query
def rewrite_query(state: GraphState):
    print("---REWRITING QUERY---")
    question = state["question"]
    
    class RewrittenQuery(BaseModel):
        optimized_query: str = Field(description="The optimized search engine query. No conversational text.")
        
    # 1. Create a parser instead of using with_structured_output
    parser = JsonOutputParser(pydantic_object=RewrittenQuery)
    
    # 2. Inject format_instructions into the prompt
    prompt = ChatPromptTemplate.from_template(
        "You are an expert web search optimizer. Your ONLY job is to look at the user's "
        "initial question and rewrite it into a highly effective search engine query. \n"
        "{format_instructions}\n"
        "Initial question: {question}"
    )
    
    # 3. Chain the parser at the end
    rewriter = prompt | llm | parser 
    
    # 4. Pass the instructions when invoking
    # Notice: The result is now a dictionary, not a Pydantic object!
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



# 1. Initialize the graph with our state
workflow = StateGraph(GraphState)

# 2. Add the nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_query", rewrite_query)

# 3. Add the edges (The Flow)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")

# Conditional Edge: After grading, do we search the web or generate?
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

# Connect the remaining straight lines
workflow.add_edge("rewrite_query", "web_search")
workflow.add_edge("generate", END)

# 4. Compile!
app = workflow.compile()


if __name__ == "__main__":
    # Test 1: Ask a question that is DEFINITELY inside your PDF
    # (Change this string to match something in your sample PDF)
    print("\n\n=== TEST 1: LOCAL KNOWLEDGE ===")
    inputs = {"question": "What is the main topic discussed in the provided document?"} 
    
    for output in app.stream(inputs):
        for key, value in output.items():
            pass # We let the nodes print their own status updates
            
    print("\n✅ Final Answer:")
    print(value["generation"])


    # Test 2: Ask a question that is DEFINITELY NOT in your PDF
    print("\n\n=== TEST 2: FORCING THE WEB FALLBACK ===")
    inputs = {"question": "What is the weather in Tokyo today?"}
    
    for output in app.stream(inputs):
        for key, value in output.items():
            pass
            
    print("\n✅ Final Answer:")
    print(value["generation"])