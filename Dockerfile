# Use official Python 3.10 slim image for a smaller footprint
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for some Python packages (like ChromaDB)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache for dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port Streamlit or your app runs on
EXPOSE 8501

# Command to run the application (Assumes you are using Streamlit for app.py)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
