import os
from pathlib import Path

import requests
from dllmforge.rag_preprocess_documents import PDFLoader, TextChunker
from dllmforge.rag_search_and_response import AzureOpenAIEmbeddingModel, IndexManager, Retriever
from tqdm import tqdm


def create_embeddings_and_index(
    file_paths: list[Path],
    index_name: str,
    chunk_size: int = 1000,
    overlap_size: int = 200,
    embedding_dim: int = 3072,
    embedding_model: str = "text-embedding-3-large",
):
    """Create embeddings for PDFs, upload them to Azure AI Search, and return the embedding model."""
    # Initialise the embedding model, pdf loader, and text chunker
    model = AzureOpenAIEmbeddingModel(model=embedding_model)
    loader = PDFLoader()
    chunker = TextChunker(chunk_size=chunk_size, overlap_size=overlap_size)

    # Embed each file and collect the embeddings
    embeddings = []
    for file_path in tqdm(file_paths, desc="Embedding files"):
        pages_with_text, file_name, _ = loader.load(file_path)
        chunks = chunker.chunk_text(pages_with_text, file_name)
        chunk_embeddings = model.embed(chunks)
        embeddings.extend(chunk_embeddings)

    # Print the number of embeddings generated for each file
    print(f"Total embeddings generated: {len(embeddings)}")

    index_manager = IndexManager(index_name=index_name, embedding_dim=embedding_dim)
    index_manager.create_index()
    index_manager.upload_documents(embeddings)

    # Print confirmation that the index has been created and documents uploaded.
    print(f"Index '{index_name}' created and documents uploaded.")

    return model


def get_retriever(index_name: str, embedding_model: str = "text-embedding-3-large"):
    """Initialise the retriever for a previously indexed collection."""
    # Initialise the embedding model and retriever for the given index.
    model = AzureOpenAIEmbeddingModel(model=embedding_model)
    retriever = Retriever(embedding_model=model, index_name=index_name)

    return retriever


def get_chunks_from_deltares_kennisbank(query: str, index_name: str):
    url = f"{os.environ['DELTARES_SEARCH_ENDPOINT']}/indexes/{index_name}/docs/search?api-version={os.environ['DELTARES_API_VERSION']}"
    headers = {"Content-Type": "application/json", "api-key": os.environ["DELTARES_SEARCH_API_KEY"]}
    body = {"search": query, "top": 10, "count": True}
    response = requests.post(url, headers=headers, json=body)
    chunks = [value["chunk"] for value in response.json()["value"]]
    return chunks
