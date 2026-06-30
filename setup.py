from setuptools import find_packages, setup

setup(
    name="crag_agent",
    version="0.2.0",
    author="Tushir Sahu",
    description="Agentic Knowledge Engine (trust-aware, temporal RAG) with MLOps Pipeline",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-huggingface",
        "lancedb",
        "langchain-google-genai",
        "fastapi",
        "uvicorn",
        "streamlit",
        "dvc",
        "networkx"
    ],
)
