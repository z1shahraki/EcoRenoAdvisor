"""
RAG retriever for document search.

Supports two modes:
1. Qdrant (if Docker is running) - production mode
2. In-memory (fallback) - simple demo mode without Docker
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
from pathlib import Path
from rag.qdrant_client import get_qdrant_client


class DocumentRetriever:
    """Retriever for semantic search over documents."""
    
    def __init__(
        self,
        collection_name: str = "renovation_docs",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        model_name: str = "BAAI/bge-small-en-v1.5",
        jsonl_path: Optional[str] = None
    ):
        """
        Initialize the document retriever.
        
        Args:
            collection_name: Qdrant collection name (for Qdrant mode)
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            model_name: Embedding model name
            jsonl_path: Path to JSONL file for in-memory mode (fallback)
        """
        self.collection_name = collection_name
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.model_name = model_name
        self.jsonl_path = jsonl_path or "data/clean/docs_text.jsonl"
        
        # Embedding model (always needed)
        self.embedder = SentenceTransformer(model_name)
        
        # Try Qdrant first
        self.client = None
        self.use_qdrant = False
        try:
            self.client = get_qdrant_client(qdrant_host, qdrant_port)
            # Test connection
            _ = self.client.get_collections()
            self.use_qdrant = True
            print(f"Using Qdrant vector database at {qdrant_host}:{qdrant_port}")
        except Exception:
            # Fallback to in-memory
            self.use_qdrant = False
            self._documents = None
            self._embeddings = None
            print(f"Qdrant not available. Using in-memory vector search (no Docker needed).")
            print(f"  Documents will be loaded from: {self.jsonl_path}")
    
    def _load_documents(self) -> None:
        """Load documents from JSONL file for in-memory search."""
        if self._documents is not None:
            return  # Already loaded
        
        self._documents = []
        
        jsonl_file = Path(self.jsonl_path)
        if not jsonl_file.exists():
            print(f"WARNING: Document file not found: {self.jsonl_path}")
            self._documents = []
            self._embeddings = np.array([])
            return
        
        print(f"Loading documents from {self.jsonl_path}...")
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text = item.get("text", "")
                    source = item.get("source", "unknown")
                    if text:
                        self._documents.append({"text": text, "source": source})
                except json.JSONDecodeError:
                    continue
        
        # Generate embeddings for all documents
        if self._documents:
            print(f"Generating embeddings for {len(self._documents)} documents...")
            texts = [doc["text"] for doc in self._documents]
            self._embeddings = self.embedder.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            print(f"Documents loaded and embedded. Ready for search.")
        else:
            self._embeddings = np.array([])
            print("No documents found in file.")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with 'text', 'source', and 'score' keys
        """
        # Generate query embedding
        query_vector = self.embedder.encode(query, normalize_embeddings=True)
        
        # Try Qdrant first (if available)
        if self.use_qdrant and self.client:
            try:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector.tolist(),
                    limit=top_k
                )
                
                results = []
                for hit in hits:
                    results.append({
                        "text": hit.payload.get("text", ""),
                        "source": hit.payload.get("source", "unknown"),
                        "score": float(hit.score)
                    })
                
                return results
            except Exception as e:
                # If Qdrant fails, fall back to in-memory
                print(f"WARNING: Qdrant search failed: {e}. Falling back to in-memory search.")
                self.use_qdrant = False
        
        # In-memory vector search (fallback or primary if Qdrant not available)
        self._load_documents()
        
        if self._documents is None or len(self._documents) == 0:
            return []
        
        # Compute cosine similarities
        similarities = np.dot(self._embeddings, query_vector)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            doc = self._documents[idx]
            results.append({
                "text": doc["text"],
                "source": doc.get("source", "unknown"),
                "score": float(similarities[idx])
            })
        
        return results
