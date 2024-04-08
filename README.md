# RAG_Langchain_Qdrant
# Injest.py and Main.py Overview

This markdown file provides an overview of the interconnected Python scripts: `injest.py` and `main.py`. These scripts work together for document ingestion, indexing, and retrieval using various libraries and APIs.

## `injest.py`

### Purpose:
- `injest.py` is responsible for ingesting documents, processing them, generating embeddings, and creating an index for retrieval.

### Key Components:
1. **Document Ingestion:** Utilizes `langchain_community.document_loaders` to load documents from a specified directory.
2. **Text Splitting:** Uses `langchain.text_splitter` for splitting documents into chunks.
3. **Embedding Model:** Loads a pre-trained embedding model from Hugging Face.
4. **Index Creation:** Utilizes the Qdrant library to create and save an index for the documents.

### Dependencies:
- `langchain_community`
- `langchain`
- `torch`
- `pymupdf`
- `qdrant-client`

## `main.py`

### Purpose:
- `main.py` retrieves documents from the index created by `injest.py`, performs similarity search based on a query, and generates responses using question answering techniques.

### Key Components:
1. **Embedding Model Loading:** Similar to `injest.py`, loads the pre-trained embedding model.
2. **Index Loading:** Loads the index created by `injest.py` using Qdrant client.
3. **Querying:** Performs similarity search based on a given query.
4. **Question Answering:** Utilizes an OpenAI model for answering questions based on retrieved documents.

### Dependencies:
- Similar to `injest.py` with additional dependencies for Q&A:
  - `openai`
  - `langchain_openai`

## Usage:
- Ensure all dependencies are installed using `requirements.txt`.
- Run `injest.py` to ingest documents, create an index, and start the Qdrant service.
- After the index is created, run `main.py` to query the index and perform question answering.

## Connection:
- `injest.py` and `main.py` are interconnected. `injest.py` creates the index using Qdrant, and `main.py` retrieves documents from this index and generates queries.

## Additional Notes:
- Both scripts demonstrate the usage of various libraries for document processing, embedding, indexing, and retrieval.
- Error handling and logging are implemented to provide insights into the execution process.

