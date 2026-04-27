from setuptools import find_packages, setup

setup(
    name="crag_agent",
    version="0.1.0",
    author="Tushir Sahu",
    description="Enterprise Agentic CRAG (Corrective RAG) System with MLOps Pipeline",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-huggingface",
        "langchain-chroma",
        "langchain-google-genai",
        "fastapi",
        "uvicorn",
        "streamlit",
        "dvc",
        "networkx"
    ],
)
