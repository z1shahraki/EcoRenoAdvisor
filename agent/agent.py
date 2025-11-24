"""
Agent logic for combining materials filtering and RAG retrieval.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from agent.tools import filter_materials, rag_search

# Default model path constant
DEFAULT_MODEL_PATH = "models/qwen2.5-3b-instruct-q4_k_m.gguf"


class RenovationAgent:
    """Agent that combines structured filtering and RAG retrieval."""
    
    def __init__(
        self,
        mode: str = "direct",
        llm_url: Optional[str] = None,
        model_name: str = "local-llm",
        model_path: Optional[str] = None
    ):
        """
        Initialize the renovation agent.
        
        Args:
            mode: "direct" (default) or "server". Direct uses llama_cpp directly, server uses HTTP API.
            llm_url: URL of the LLM server (only used in server mode)
            model_name: Model name to use in API calls (server mode)
            model_path: Path to GGUF model file (direct mode). Auto-detected if not provided.
        """
        self.mode = mode
        self.llm_url = llm_url or os.getenv("LLM_URL", "http://localhost:8000/v1/chat/completions")
        self.model_name = model_name
        self.model_path = model_path or self._find_model_path()
        self._llm_model = None
        
    def _find_model_path(self) -> Optional[str]:
        """Find model file automatically."""
        # Check environment variable first
        env_path = os.getenv("LLM_MODEL_PATH")
        if env_path and Path(env_path).exists():
            return env_path
        
        # Check default location
        default_path = Path(DEFAULT_MODEL_PATH)
        if default_path.exists():
            return str(default_path)
        
        # Check common alternatives
        alt_paths = [
            "models/llama-3.2-3b-instruct-q4_k_m.gguf",
            "models/qwen2.5-3b-instruct-q4_k_m.gguf",
        ]
        for path in alt_paths:
            if Path(path).exists():
                return path
        
        return None
        
    def _get_direct_llm(self):
        """Get direct llama_cpp model if available."""
        if self._llm_model is not None:
            return self._llm_model
            
        if not self.model_path:
            return None
            
        if not Path(self.model_path).exists():
            return None
            
        try:
            from llama_cpp import Llama
            self._llm_model = Llama(
                model_path=str(self.model_path),
                n_ctx=2048,
                n_threads=8,
                verbose=False
            )
            return self._llm_model
        except ImportError:
            return None
        except Exception as e:
            print(f"Warning: Could not load model directly: {e}")
            return None
    
    def call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Call the local LLM.
        
        In direct mode: Uses llama_cpp.Llama directly.
        In server mode: Uses HTTP API to llama_cpp.server.
        
        Args:
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Direct mode (default)
        if self.mode == "direct":
            llm = self._get_direct_llm()
            if llm is None:
                if not self.model_path:
                    return "Error: No model file found. Please download a model to models/ directory or set LLM_MODEL_PATH environment variable."
                if not Path(self.model_path).exists():
                    return f"Error: Model file not found at {self.model_path}. Please download the model or check the path."
                return "Error: Could not load model. Make sure llama-cpp-python is installed: pip install llama-cpp-python"
            
            try:
                # Use chat format for instruct models
                response = llm(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    stop=["\n\n", "User:", "Assistant:"],
                    echo=False
                )
                
                # Handle different response formats
                if isinstance(response, dict):
                    if "choices" in response and len(response["choices"]) > 0:
                        choice = response["choices"][0]
                        # Handle both completion and chat formats
                        if "text" in choice:
                            return choice["text"].strip()
                        elif "message" in choice and "content" in choice["message"]:
                            return choice["message"]["content"].strip()
                    # Fallback: try to extract any text field
                    text = response.get("text", str(response))
                    return str(text).strip()
                
                # If response is string directly
                return str(response).strip()
                
            except Exception as e:
                return f"Error calling LLM: {e}"
        
        # Server mode (optional)
        elif self.mode == "server":
            try:
                import requests
                response = requests.post(
                    self.llm_url,
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "temperature": 0.7
                    },
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except ImportError:
                return "Error: requests library not available. Install with: pip install requests"
            except requests.exceptions.RequestException as e:
                if "Connection" in str(e) or "refused" in str(e).lower():
                    return f"Error: LLM server not reachable at {self.llm_url}. Start the server with: python -m llama_cpp.server --model models/your-model.gguf --port 8000"
                return f"Error calling LLM server: {e}"
            except (KeyError, IndexError) as e:
                return f"Error parsing LLM response: {e}"
        
        else:
            return f"Error: Invalid mode '{self.mode}'. Use 'direct' or 'server'."
    
    def agent(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k_materials: int = 5,
        top_k_docs: int = 3
    ) -> str:
        """
        Process a user query using materials filtering and RAG retrieval.
        
        Args:
            query: User question
            filters: Dictionary with filter parameters (category, max_price, min_eco, voc)
            top_k_materials: Number of materials to retrieve
            top_k_docs: Number of document chunks to retrieve
            
        Returns:
            Agent response
        """
        filters = filters or {}
        
        # Get filtered materials
        materials = filter_materials(
            category=filters.get("category"),
            max_price=filters.get("max_price"),
            min_eco=filters.get("min_eco"),
            voc=filters.get("voc")
        )
        
        # Get relevant documents
        docs = rag_search(query, top_k=top_k_docs)
        
        # Build prompt
        materials_str = json.dumps(materials, indent=2) if materials else "No materials found matching the criteria."
        docs_str = "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(docs)]) if docs else "No relevant documents found."
        
        prompt = f"""You are a helpful assistant for sustainable renovation planning. Answer the user's question based on the provided materials and documents.

User question: {query}

Candidate materials from database:
{materials_str}

Relevant documents:
{docs_str}

Please provide a clear, helpful recommendation that:
1. Addresses the user's specific question
2. References specific materials when relevant
3. Incorporates information from the documents
4. Considers sustainability and eco-friendliness
5. Is practical and actionable

Response:"""
        
        return self.call_llm(prompt)


# Convenience function
def agent(query: str, filters: Optional[Dict[str, Any]] = None, mode: str = "direct") -> str:
    """
    Convenience function for agent processing.
    
    Args:
        query: User question
        filters: Filter parameters
        mode: "direct" (default) or "server"
        
    Returns:
        Agent response
    """
    agent_instance = RenovationAgent(mode=mode)
    return agent_instance.agent(query, filters)
