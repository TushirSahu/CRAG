import os
from dotenv import load_dotenv

load_dotenv(override=True)

print("TRACING:", os.getenv("LANGCHAIN_TRACING_V2"))
print("PROJECT:", os.getenv("LANGCHAIN_PROJECT"))

from langchain_google_genai import ChatGoogleGenerativeAI
# Using Gemma since your Gemini Free Tier quota is exhausted for today!
llm = ChatGoogleGenerativeAI(model="gemma-3-4b-it")

print("Invoking LLM...")
try:
    print(llm.invoke("Say 'Hello LangSmith! The telemetry is working.'"))
    print("Done. Check dashboard.")
except Exception as e:
    print("Error:", e)
