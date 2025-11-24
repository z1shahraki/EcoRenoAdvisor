"""
Integration test: LLM + simple RAG scenario (direct mode).

This test verifies that:
1. A local GGUF model is available.
2. Embeddings can be used to find relevant context.
3. The RenovationAgent can call the LLM in direct mode and return a sensible answer.

If the local model file is not present, the test is skipped.
"""

import pytest
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from agent.agent import RenovationAgent

# Path to the local GGUF model used in direct mode
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"


def has_local_model() -> bool:
    """Return True if the local GGUF model file exists."""
    return MODEL_PATH.exists()


# Module-level check: if no model, skip the whole file
if not has_local_model():
    pytest.skip(
        f"Local LLM model not found at {MODEL_PATH}. "
        "This integration test requires the GGUF model in models/.",
        allow_module_level=True,
    )


@pytest.mark.integration
def test_llm_with_simple_rag():
    """
    Test LLM with a simple RAG scenario:
    1. Create embeddings for sample documents
    2. Find most relevant document for a query
    3. Call LLM with the query and context (direct mode)
    4. Verify LLM responds appropriately
    """
    documents = [
        "Low VOC water-based paint is ideal for kids bedrooms because it has minimal harmful emissions.",
        "Ceramic tiles are durable and easy to clean, making them perfect for kitchen floors.",
        "Wool carpet provides natural insulation and is eco-friendly for living rooms.",
    ]

    query = "What paint should I use for my child's bedroom?"

    # Step 1: embeddings
    print("\n[Step 1] Loading embedding model...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    print("[Step 2] Encoding documents and query...")
    doc_embeddings = model.encode(documents, normalize_embeddings=True)
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    similarities = doc_embeddings @ query_embedding
    top_doc_idx = int(np.argmax(similarities))
    relevant_context = documents[top_doc_idx]

    print(f"[Step 3] Most relevant document: {relevant_context[:60]}...")
    print(f"[Step 4] Similarity score: {similarities[top_doc_idx]:.3f}")

    # Basic sanity checks on retrieval
    assert top_doc_idx == 0, f"Expected paint document at index 0, got {top_doc_idx}"
    assert "paint" in relevant_context.lower()

    # Step 2: call LLM in direct mode
    print("[Step 5] Calling LLM in direct mode...")
    agent = RenovationAgent(mode="direct")
    prompt = f"""You are a helpful renovation assistant. Answer the user's question based on the provided context.

Context: {relevant_context}

User question: {query}

Please provide a brief, helpful answer (2–3 sentences):"""

    response = agent.call_llm(prompt, max_tokens=100)
    print(f"[Step 6] LLM Response: {response[:200]}...")

    # Verify response looks healthy
    assert isinstance(response, str)
    assert len(response.strip()) > 10
    assert not response.startswith("Error")

    lower = response.lower()
    relevant_terms = ["paint", "bedroom", "voc", "child", "kid"]
    assert any(t in lower for t in relevant_terms), (
        f"Response should mention at least one relevant term. Got: {response[:120]}"
    )
    
    print("\n[PASS] LLM RAG test passed! Full RAG + LLM flow works in direct mode.")


@pytest.mark.integration
def test_llm_basic_functionality():
    """Basic direct-mode LLM sanity check without RAG."""
    print("\n[Test] Basic LLM functionality (direct mode)...")
    agent = RenovationAgent(mode="direct")

    prompt = "Say: Hello, I am working correctly, in one short sentence."
    response = agent.call_llm(prompt, max_tokens=50)
    
    print(f"LLM Response: {response}")

    assert isinstance(response, str)
    assert len(response.strip()) > 5
    assert not response.startswith("Error")
    
    print("[PASS] Basic LLM test passed!")
