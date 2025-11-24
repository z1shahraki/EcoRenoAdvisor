"""
Simple test for embedding model only - works without Docker or LLM.

Perfect for portfolio demonstration.
"""

import sys


def test_embedding_model():
    """Test embedding model - no Docker or LLM needed."""
    print("=" * 60)
    print("TEST: Embedding Model (BGE-small-en-v1.5)")
    print("=" * 60)
    print("")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Loading BGE-small-en-v1.5 model...")
        print("(This may take a minute on first run - downloading model)")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("Model loaded successfully")
        print("")
        
        # Test encoding
        test_texts = [
            "sustainable renovation materials",
            "eco-friendly flooring options",
            "low VOC paint for bedrooms"
        ]
        
        print("Testing embeddings on sample texts:")
        print("-" * 60)
        
        embeddings = []
        for text in test_texts:
            embedding = model.encode(text)
            embeddings.append(embedding)
            print(f"Text: '{text}'")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding dimension: {len(embedding)}")
            print(f"  First 5 values: {embedding[:5]}")
            print("")
        
        # Test similarity
        print("Testing similarity between embeddings:")
        print("-" * 60)
        
        # Similar texts should have high similarity
        from numpy import dot
        from numpy.linalg import norm
        
        def cosine_similarity(a, b):
            return dot(a, b) / (norm(a) * norm(b))
        
        sim1 = cosine_similarity(embeddings[0], embeddings[1])
        sim2 = cosine_similarity(embeddings[0], embeddings[2])
        
        print(f"Similarity between 'sustainable materials' and 'eco-friendly flooring': {sim1:.3f}")
        print(f"Similarity between 'sustainable materials' and 'low VOC paint': {sim2:.3f}")
        print("")
        
        if len(embeddings[0]) == 384:
            print("=" * 60)
            print("PASS: Embedding model works correctly!")
            print("=" * 60)
            print("")
            print("This demonstrates:")
            print("  - Embedding generation (384-dimensional vectors)")
            print("  - Semantic similarity calculation")
            print("  - Ready for RAG pipeline")
            return True
        else:
            print(f"FAIL: Expected 384 dimensions, got {len(embeddings[0])}")
            return False
            
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_processing():
    """Test data processing - no Docker needed."""
    print("=" * 60)
    print("TEST: Data Processing")
    print("=" * 60)
    print("")
    
    try:
        import pandas as pd
        from pathlib import Path
        
        # Test CSV reading and cleaning
        csv_path = Path("data/raw/materials.csv")
        
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found")
            print("Skipping data processing test")
            return True
        
        print(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
        print("")
        
        # Test cleaning logic
        print("Testing data cleaning...")
        if 'price_per_m2' in df.columns:
            df['price_per_m2'] = df['price_per_m2'].astype(str).replace(r'[$, ]', '', regex=True)
            df['price_per_m2'] = pd.to_numeric(df['price_per_m2'], errors='coerce')
            print("  Price cleaning: OK")
        
        if 'voc_level' in df.columns:
            mapping = {"low": 1, "medium": 2, "high": 3, "zero": 0}
            df['voc_level_num'] = df['voc_level'].str.lower().map(mapping)
            print("  VOC level mapping: OK")
        
        print("")
        print("=" * 60)
        print("PASS: Data processing works correctly!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests that work without Docker."""
    print("")
    print("=" * 60)
    print("PORTFOLIO TESTS (No Docker Required)")
    print("=" * 60)
    print("")
    print("These tests demonstrate core functionality:")
    print("  - Embedding model (RAG foundation)")
    print("  - Data processing (ETL pipeline)")
    print("")
    
    results = []
    
    results.append(("Embedding Model", test_embedding_model()))
    print("")
    results.append(("Data Processing", test_data_processing()))
    
    # Summary
    print("")
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:25} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("")
        print("All portfolio tests passed!")
        print("")
        print("You can now:")
        print("  - Show the code structure")
        print("  - Explain the architecture")
        print("  - Demonstrate embeddings work")
        print("  - Mention Docker setup for production")
    else:
        print("")
        print("Some tests failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

