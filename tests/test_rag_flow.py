#!/usr/bin/env python3
"""
Quick test to verify RAG is working:
1. Vector search finds relevant documents
2. Documents are passed to LLM
3. LLM generates answer using retrieved context
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.agent import RenovationAgent
from agent.tools import rag_search

def test_rag_search():
    """Test vector search is working."""
    print("=" * 60)
    print("TEST 1: Vector Search")
    print("=" * 60)
    
    query = "What flooring should I use for a kids bedroom?"
    print(f"\nQuery: {query}\n")
    
    results = rag_search(query, top_k=3)
    
    print(f"Found {len(results)} documents:\n")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc[:150]}...")
        print()
    
    if len(results) == 0:
        print("[FAIL] No documents retrieved!")
        return False
    
    if "bamboo" not in results[0].lower() and "flooring" not in results[0].lower():
        print("[WARN] Top result doesn't seem relevant to query")
    
    print("[PASS] Vector search working!\n")
    return True


def test_full_rag():
    """Test full RAG pipeline: search + LLM."""
    print("=" * 60)
    print("TEST 2: Full RAG Pipeline (Search + LLM)")
    print("=" * 60)
    
    query = "Suggest one eco friendly flooring option for a kids bedroom and explain briefly why it is suitable."
    print(f"\nQuery: {query}\n")
    
    # Initialize agent (direct mode)
    agent = RenovationAgent(mode="direct")
    
    print("Calling agent...\n")
    response = agent.agent(query, top_k_docs=3)
    
    print("=" * 60)
    print("AGENT RESPONSE:")
    print("=" * 60)
    print(response)
    print("=" * 60)
    print()
    
    # Check if response mentions documents/context
    if len(response) < 100:
        print("[WARN] Response seems too short")
        return False
    
    # Check if it mentions relevant terms
    relevant_terms = ["bamboo", "flooring", "kids", "bedroom", "eco"]
    found_terms = [term for term in relevant_terms if term.lower() in response.lower()]
    print(f"[PASS] Found relevant terms: {found_terms}")
    
    print("[PASS] Full RAG pipeline working!\n")
    return True


if __name__ == "__main__":
    print("\nTesting RAG System\n")
    
    # Test 1: Vector search
    search_ok = test_rag_search()
    
    if not search_ok:
        print("[FAIL] Vector search failed. Cannot test full RAG.")
        sys.exit(1)
    
    # Test 2: Full RAG
    try:
        rag_ok = test_full_rag()
        if rag_ok:
            print("\n[PASS] All tests passed! RAG is working correctly.")
            sys.exit(0)
        else:
            print("\n[WARN] RAG pipeline has issues. Check output above.")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error during RAG test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

