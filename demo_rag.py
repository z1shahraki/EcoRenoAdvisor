"""
Demo script: Run a simple RAG + LLM example.

This demonstrates the full RAG + LLM flow using direct mode.
Run with: python demo_rag.py
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from agent.agent import RenovationAgent


def main():
    """Run a simple RAG + LLM demonstration."""
    
    # Check model exists
    model_path = Path("models/qwen2.5-3b-instruct-q4_k_m.gguf")
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Please download qwen2.5-3b-instruct-q4_k_m.gguf to models/ directory")
        return
    
    print("=" * 60)
    print("EcoRenoAdvisor - RAG + LLM Demo")
    print("=" * 60)
    print()
    
    # Sample documents
    documents = [
        "Low VOC water-based paint is ideal for kids bedrooms because it has minimal harmful emissions.",
        "Ceramic tiles are durable and easy to clean, making them perfect for kitchen floors.",
        "Wool carpet provides natural insulation and is eco-friendly for living rooms.",
    ]
    
    query = "What paint should I use for my child's bedroom?"
    
    print(f"Question: {query}")
    print()
    
    # Step 1: Load embeddings
    print("[1/4] Loading embedding model...")
    embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    
    # Step 2: Find relevant context
    print("[2/4] Finding relevant document...")
    doc_embeddings = embedding_model.encode(documents, normalize_embeddings=True)
    query_embedding = embedding_model.encode([query], normalize_embeddings=True)[0]
    
    similarities = doc_embeddings @ query_embedding
    top_doc_idx = int(np.argmax(similarities))
    relevant_context = documents[top_doc_idx]
    
    print(f"    → Selected document {top_doc_idx + 1}: {relevant_context[:70]}...")
    print(f"    → Similarity score: {similarities[top_doc_idx]:.3f}")
    print()
    
    # Step 3: Call LLM in direct mode
    print("[3/4] Loading LLM model (direct mode)...")
    agent = RenovationAgent(mode="direct")
    
    prompt = f"""You are a helpful renovation assistant. Answer the user's question based on the provided context.

Context: {relevant_context}

User question: {query}

Please provide a brief, helpful answer (2-3 sentences):"""
    
    print("[4/4] Generating response...")
    response = agent.call_llm(prompt, max_tokens=100)
    print()
    
    # Display result
    print("=" * 60)
    print("Answer:")
    print("=" * 60)
    print(response)
    print()
    
    # Check if it worked
    if response.startswith("Error"):
        print("[ERROR] Error occurred - check model file and dependencies")
    else:
        print("[PASS] RAG + LLM flow completed successfully!")


if __name__ == "__main__":
    main()

