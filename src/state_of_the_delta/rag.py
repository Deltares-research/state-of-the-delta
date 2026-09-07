from pathlib import Path

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
