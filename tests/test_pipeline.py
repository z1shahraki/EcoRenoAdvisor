"""
Test script to verify the pipeline is working correctly.

Run this after setup to test each component.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import pandas as pd
        import pypdf
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        import gradio as gr
        import requests
        print("All imports successful")
        return True
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False


def test_data_files():
    """Test that data files exist."""
    print("\nTesting data files...")
    materials_csv = Path("data/raw/materials.csv")
    if materials_csv.exists():
        print(f"Found {materials_csv}")
    else:
        print(f"WARNING: {materials_csv} not found (this is okay if you haven't added data yet)")
    
    return True


def test_qdrant():
    """Test Qdrant connection."""
    print("\nTesting Qdrant connection...")
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        print("Qdrant connection successful")
        print(f"  Collections: {[c.name for c in collections.collections]}")
        return True
    except Exception as e:
        print(f"ERROR: Qdrant connection failed: {e}")
        print("  Make sure Qdrant is running: docker-compose up -d")
        return False


def test_llm_server():
    """Test LLM server connection."""
    print("\nTesting LLM server...")
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("LLM server is running")
            return True
        else:
            print(f"WARNING: LLM server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("WARNING: LLM server not running")
        print("  Start it with: python -m llama_cpp.server --model models/your-model.gguf --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"ERROR: Error checking LLM server: {e}")
        return False


def test_ingestion():
    """Test ingestion scripts."""
    print("\nTesting ingestion scripts...")
    
    # Test materials cleaning
    try:
        from ingestion.clean_materials import clean_materials
        materials_csv = Path("data/raw/materials.csv")
        if materials_csv.exists():
            clean_materials(str(materials_csv), "data/clean/materials_test.parquet")
            if Path("data/clean/materials_test.parquet").exists():
                print("Materials cleaning works")
                Path("data/clean/materials_test.parquet").unlink()  # Clean up
            else:
                print("ERROR: Materials cleaning failed")
                return False
        else:
            print("WARNING: Skipping materials test (no CSV file)")
    except Exception as e:
        print(f"ERROR: Materials cleaning error: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Testing EcoRenoAdvisor Pipeline\n")
    print("=" * 50)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Data Files", test_data_files()))
    results.append(("Qdrant", test_qdrant()))
    results.append(("LLM Server", test_llm_server()))
    results.append(("Ingestion", test_ingestion()))
    
    print("\n" + "=" * 50)
    print("\nTest Summary:")
    print("-" * 50)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:20} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nWARNING: Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()

