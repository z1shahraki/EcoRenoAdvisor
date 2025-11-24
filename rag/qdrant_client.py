"""
Qdrant client utilities for vector database operations.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from typing import Optional


def get_qdrant_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    """
    Create and return a Qdrant client instance.
    
    Args:
        host: Qdrant server host
        port: Qdrant server port
        
    Returns:
        QdrantClient instance
    """
    return QdrantClient(host=host, port=port)


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int = 384
) -> None:
    """
    Ensure a collection exists, create if it doesn't.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        vector_size: Size of embedding vectors
    """
    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if collection_name not in collection_names:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
            print(f"Created collection: {collection_name}")
        else:
            print(f"Collection already exists: {collection_name}")
    except Exception as e:
        print(f"WARNING: Error checking/creating collection: {e}")

