"""
Agent tools for materials filtering and document retrieval.
"""

import pandas as pd
from typing import Optional, List, Dict, Any
from rag.retriever import DocumentRetriever
from pathlib import Path


class MaterialsFilter:
    """Tool for filtering materials based on various criteria."""
    
    def __init__(self, materials_path: str = "data/clean/materials.parquet"):
        """
        Initialize materials filter.
        
        Args:
            materials_path: Path to cleaned materials Parquet file
        """
        self.materials_path = materials_path
        self._materials = None
    
    @property
    def materials(self) -> pd.DataFrame:
        """Lazy load materials dataframe."""
        if self._materials is None:
            if not Path(self.materials_path).exists():
                print(f"WARNING: Materials file not found: {self.materials_path}")
                self._materials = pd.DataFrame()
            else:
                self._materials = pd.read_parquet(self.materials_path)
        return self._materials
    
    def filter_materials(
        self,
        category: Optional[str] = None,
        max_price: Optional[float] = None,
        min_eco: Optional[float] = None,
        voc: Optional[int] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Filter materials based on criteria.
        
        Args:
            category: Category filter (substring match, case-insensitive)
            max_price: Maximum price per m2
            min_eco: Minimum eco score
            voc: Maximum VOC level (0=zero, 1=low, 2=medium, 3=high)
            limit: Maximum number of results
            
        Returns:
            List of material dictionaries
        """
        df = self.materials.copy()
        
        if df.empty:
            return []
        
        # Apply filters
        if category:
            df = df[df['category'].str.contains(category, case=False, na=False)]
        
        if max_price is not None:
            df = df[df['price_per_m2'] <= max_price]
        
        if min_eco is not None:
            df = df[df['eco_score'] >= min_eco]
        
        if voc is not None:
            df = df[df['voc_level_num'] <= voc]
        
        # Return top results
        return df.head(limit).to_dict(orient="records")


class RAGSearchTool:
    """Tool for RAG-based document search."""
    
    def __init__(
        self,
        collection_name: str = "renovation_docs",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333
    ):
        """
        Initialize RAG search tool.
        
        Args:
            collection_name: Qdrant collection name
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
        """
        self.retriever = DocumentRetriever(
            collection_name=collection_name,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port
        )
    
    def search(self, question: str, top_k: int = 3) -> List[str]:
        """
        Search for relevant documents.
        
        Args:
            question: Search query
            top_k: Number of results
            
        Returns:
            List of document text snippets
        """
        results = self.retriever.search(question, top_k=top_k)
        return [r["text"] for r in results]


# Convenience functions for agent use
def filter_materials(
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    min_eco: Optional[float] = None,
    voc: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Filter materials - convenience function."""
    filter_tool = MaterialsFilter()
    return filter_tool.filter_materials(
        category=category,
        max_price=max_price,
        min_eco=min_eco,
        voc=voc
    )


def rag_search(question: str, top_k: int = 3) -> List[str]:
    """RAG search - convenience function."""
    search_tool = RAGSearchTool()
    return search_tool.search(question, top_k=top_k)

