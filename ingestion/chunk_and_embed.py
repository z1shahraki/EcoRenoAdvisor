"""
Chunk documents, generate embeddings, and upload to Qdrant.

This script:
- Loads extracted documents from JSONL
- Chunks text into overlapping segments
- Generates embeddings using BGE-small-en-v1.5
- Uploads to Qdrant vector database
"""

import json
import hashlib
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Input text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def create_collection(client: QdrantClient, collection_name: str, vector_size: int = 384) -> None:
    """
    Create or recreate Qdrant collection.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        vector_size: Size of embedding vectors
    """
    try:
        # Try to delete existing collection
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        pass
    
    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    print(f"Created collection: {collection_name}")


def ingest_documents(
    jsonl_path: str,
    collection_name: str = "renovation_docs",
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    chunk_size: int = 500,
    overlap: int = 50
) -> None:
    """
    Ingest documents into Qdrant.
    
    Args:
        jsonl_path: Path to JSONL file with documents
        collection_name: Qdrant collection name
        qdrant_host: Qdrant server host
        qdrant_port: Qdrant server port
        chunk_size: Words per chunk
        overlap: Words to overlap between chunks
    """
    # Initialize embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("Model loaded")
    
    # Initialize Qdrant client
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    
    # Create collection
    create_collection(client, collection_name, vector_size=384)
    
    # Process documents
    if not Path(jsonl_path).exists():
        print(f"WARNING: File not found: {jsonl_path}")
        return
    
    total_chunks = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                text = item.get("text", "")
                source = item.get("source", "unknown")
                
                if not text:
                    continue
                
                # Chunk the text
                chunks = chunk_text(text, chunk_size, overlap)
                
                # Generate embeddings and upload
                points = []
                for chunk in chunks:
                    # Generate embedding
                    vec = model.encode(chunk).tolist()
                    
                    # Create unique ID from chunk content
                    chunk_id = int(hashlib.md5(chunk.encode()).hexdigest()[:8], 16)
                    
                    points.append(
                        PointStruct(
                            id=chunk_id,
                            vector=vec,
                            payload={
                                "source": source,
                                "text": chunk
                            }
                        )
                    )
                
                # Batch upsert
                if points:
                    client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    total_chunks += len(points)
                    print(f"  Processed {source}: {len(points)} chunks")
            
            except json.JSONDecodeError as e:
                print(f"  ERROR: Error parsing line {line_num}: {e}")
            except Exception as e:
                print(f"  ERROR: Error processing line {line_num}: {e}")
    
    print(f"\nIngestion complete. Total chunks: {total_chunks}")


if __name__ == "__main__":
    jsonl_path = "data/clean/docs_text.jsonl"
    
    ingest_documents(jsonl_path)

