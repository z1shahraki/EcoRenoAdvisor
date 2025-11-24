"""
Simple test to verify basic components work independently:
1. Embedding model
2. Vector database (Qdrant)
3. LLM model

Run this BEFORE using the full pipeline to ensure everything works.
"""

import sys
from pathlib import Path


def test_embedding_model():
    """Test 1: Verify embedding model can generate vectors."""
    print("=" * 60)
    print("TEST 1: Embedding Model")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        
        print("Loading BGE-small-en-v1.5 model...")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("Model loaded successfully")
        
        # Test encoding
        test_text = "sustainable renovation materials"
        print(f"Encoding test text: '{test_text}'")
        embedding = model.encode(test_text)
        
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
        if len(embedding) == 384:
            print("PASS: Embedding model works correctly")
            return True
        else:
            print(f"FAIL: Expected 384 dimensions, got {len(embedding)}")
            return False
            
    except Exception as e:
        print(f"FAIL: Error testing embedding model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_database():
    """Test 2: Verify Qdrant connection and basic operations."""
    print("\n" + "=" * 60)
    print("TEST 2: Vector Database (Qdrant)")
    print("=" * 60)
    
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        print("Connecting to Qdrant at localhost:6333...")
        client = QdrantClient(host="localhost", port=6333)
        
        # Test connection
        collections = client.get_collections()
        print(f"Connected successfully. Found {len(collections.collections)} collections")
        
        # Create test collection
        test_collection = "test_basic_components"
        print(f"\nCreating test collection: {test_collection}")
        
        try:
            client.delete_collection(test_collection)
            print("Deleted existing test collection")
        except:
            pass
        
        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print("Test collection created")
        
        # Insert test vector
        print("\nInserting test vector...")
        test_vector = [0.1] * 384  # Simple test vector
        client.upsert(
            collection_name=test_collection,
            points=[PointStruct(
                id=1,
                vector=test_vector,
                payload={"text": "test document"}
            )]
        )
        print("Vector inserted")
        
        # Search test
        print("\nTesting search...")
        results = client.search(
            collection_name=test_collection,
            query_vector=test_vector,
            limit=1
        )
        
        if results and len(results) > 0:
            print(f"Search successful. Found {len(results)} result(s)")
            print(f"Result text: {results[0].payload.get('text')}")
            print("PASS: Vector database works correctly")
            
            # Cleanup
            client.delete_collection(test_collection)
            print("Cleaned up test collection")
            return True
        else:
            print("FAIL: Search returned no results")
            return False
            
    except Exception as e:
        print(f"FAIL: Error testing vector database: {e}")
        print("Make sure Qdrant is running: docker-compose up -d")
        import traceback
        traceback.print_exc()
        return False


def test_llm_server():
    """Test 3: Verify LLM server is accessible and can generate text."""
    print("\n" + "=" * 60)
    print("TEST 3: LLM Server")
    print("=" * 60)
    
    try:
        import requests
        
        # Check health endpoint
        print("Checking LLM server health...")
        try:
            health_response = requests.get("http://localhost:8000/health", timeout=2)
            if health_response.status_code == 200:
                print("Server health check passed")
            else:
                print(f"Health check returned status {health_response.status_code}")
        except:
            print("Health endpoint not available (this is okay)")
        
        # Test generation
        print("\nTesting text generation...")
        test_prompt = "Say hello in one sentence."
        
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "local-llm",
                "messages": [{"role": "user", "content": test_prompt}],
                "max_tokens": 50,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["message"]["content"]
            print(f"Generated text: {generated_text}")
            print("PASS: LLM server works correctly")
            return True
        else:
            print(f"FAIL: Server returned status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("FAIL: Cannot connect to LLM server")
        print("Make sure LLM server is running:")
        print("  python -m llama_cpp.server --model models/your-model.gguf --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print(f"FAIL: Error testing LLM server: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_simple():
    """Test 4: Simple end-to-end test with embedding + vector DB."""
    print("\n" + "=" * 60)
    print("TEST 4: End-to-End (Embedding + Vector DB)")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        # Load embedding model
        print("Loading embedding model...")
        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        # Connect to Qdrant
        print("Connecting to Qdrant...")
        client = QdrantClient(host="localhost", port=6333)
        
        # Create test collection
        test_collection = "test_e2e"
        try:
            client.delete_collection(test_collection)
        except:
            pass
        
        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print("Test collection created")
        
        # Create test documents
        test_docs = [
            "Bamboo flooring is sustainable and eco-friendly",
            "Wool insulation provides excellent thermal properties",
            "Low VOC paint is better for indoor air quality"
        ]
        
        print(f"\nEmbedding {len(test_docs)} test documents...")
        points = []
        for i, doc in enumerate(test_docs):
            vector = embedder.encode(doc).tolist()
            points.append(PointStruct(
                id=i + 1,
                vector=vector,
                payload={"text": doc, "id": i + 1}
            ))
        
        # Insert into Qdrant
        client.upsert(collection_name=test_collection, points=points)
        print("Documents inserted into vector database")
        
        # Search
        query = "eco-friendly flooring options"
        print(f"\nSearching for: '{query}'")
        query_vector = embedder.encode(query).tolist()
        
        results = client.search(
            collection_name=test_collection,
            query_vector=query_vector,
            limit=2
        )
        
        if results and len(results) > 0:
            print(f"Found {len(results)} relevant document(s):")
            for i, result in enumerate(results, 1):
                print(f"  {i}. Score: {result.score:.3f}")
                print(f"     Text: {result.payload['text']}")
            
            print("PASS: End-to-end test successful")
            
            # Cleanup
            client.delete_collection(test_collection)
            return True
        else:
            print("FAIL: No results found")
            return False
            
    except Exception as e:
        print(f"FAIL: Error in end-to-end test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all basic component tests."""
    print("\n" + "=" * 60)
    print("BASIC COMPONENT TESTS")
    print("Testing: Embeddings, Vector DB, LLM")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Embedding model
    results.append(("Embedding Model", test_embedding_model()))
    
    # Test 2: Vector database
    results.append(("Vector Database", test_vector_database()))
    
    # Test 3: LLM server (optional - skip if not running)
    results.append(("LLM Server", test_llm_server()))
    
    # Test 4: End-to-end
    results.append(("End-to-End", test_end_to_end_simple()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name:25} {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\nAll basic components are working!")
        print("You can now proceed with the full pipeline.")
    else:
        print("\nSome tests failed. Fix the issues before proceeding.")
        print("\nCommon fixes:")
        print("  - Embedding model: Will download automatically on first use")
        print("  - Vector DB: Run 'docker-compose up -d'")
        print("  - LLM Server: Start with 'python -m llama_cpp.server --model models/your-model.gguf --host 0.0.0.0 --port 8000'")
        sys.exit(1)


if __name__ == "__main__":
    main()

