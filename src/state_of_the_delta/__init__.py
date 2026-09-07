import importlib

from .memo import Memo
from .rag import create_embeddings_and_index, get_retriever

__version__ = importlib.metadata.version(__package__)

__all__ = ["Memo", "__version__", "create_embeddings_and_index", "get_retriever"]
