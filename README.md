# RAG Application with AWS Bedrock

A Retrieval-Augmented Generation (RAG) application built with LangChain and AWS Bedrock.

## Overview

This application implements a RAG pipeline that:
1. Processes documents (PDFs, etc.)
2. Stores embeddings in ChromaDB
3. Retrieves relevant context based on user queries
4. Generates responses using AWS Bedrock models

## Prerequisites

- Python 3.11+
- AWS account with Bedrock access
- AWS credentials configured

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Environment Variables

Create a `.env` file with the following variables:
```
AWS_REGION=us-east-1
CHROMA_PATH=data/chroma
IS_USING_IMAGE_RUNTIME=False
```

## Usage

### Running Locally

```bash
python src/app_api_handler.py
```

The API will be available at http://localhost:8000

### Docker Deployment

Build and run the Docker container:
```bash
docker build --platform linux/amd64 -t aws_rag_app .
docker run --platform linux/amd64 --rm -it \
    -p 8000:8000 \
    --env-file .env \
    aws_rag_app
```

### Testing AWS Credentials

```bash
python test_aws_credentials.py
```

## Project Structure

```
rag-app/
├── data/
│   └── chroma/         # ChromaDB storage
├── src/
│   ├── rag_app/
│   │   ├── get_chroma_db.py        # ChromaDB initialization
│   │   ├── get_embeddings.py       # Embedding function setup
│   │   └── ...
│   ├── app_api_handler.py          # FastAPI application
│   └── ...
├── .env                 # Environment variables
├── .gitignore           # Git ignore file
├── Dockerfile           # Docker configuration
├── pyproject.toml       # Project metadata and dependencies
└── README.md            # This file
```

## License

[Specify your license here]

