"""
Mini LLM experiment runner

Goal:
    - Load a small JSONL dataset with fields: prompt, expected
    - Call a model client to get predictions
    - Compute simple metrics like exact match and length difference
    - Print a small experiment report

This file is only for testing GitHub Copilot intelligence,
not for production use.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import statistics
from utils import clean_text


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Example:
    prompt: str
    expected: str
    prediction: Optional[str] = None


# ----------------------------
# Dataset utilities
# ----------------------------

def create_sample_dataset(path: Path) -> None:
    """
    Create a small sample JSONL dataset if it does not exist.
    Each line has keys: prompt, expected.
    """
    sample = [
        {
            "prompt": "Summarise: Data science is the art of turning data into decisions.",
            "expected": "Data science turns data into decisions."
        },
        {
            "prompt": "Rewrite in simpler English: Retrieval augmented generation helps models use external knowledge.",
            "expected": "Retrieval augmented generation helps models use outside knowledge."
        },
        {
            "prompt": "Short definition: What is a language model",
            "expected": "A language model predicts the next word in a sequence of text."
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as f:
        for row in sample:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset(path: str) -> List[Example]:
    """
    Load a JSONL dataset from disk.
    Each line is a JSON object with keys: prompt, expected.
    If the file does not exist, a small sample dataset is created.
    """
    p = Path(path)

    if not p.exists():
        print(f"[info] dataset not found at {p}, creating a sample dataset.")
        create_sample_dataset(p)

    examples: List[Example] = []
    with p.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            examples.append(
                Example(
                    prompt=obj["prompt"],
                    expected=obj["expected"],
                )
            )
    return examples


# ----------------------------
# Dummy LLM client
# ----------------------------

class LLMClient:
    """
    Very small abstraction layer around an LLM.

    In real life this could call OpenAI, Azure, or a local model.
    For this test we return a simple transformation of the prompt
    so that the pipeline can be tested without external keys.
    """

    def __init__(self, model_name: str = "stub"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """
        Fake generation method.

        Strategy:
            - Take the prompt
            - Keep only the first sentence or up to 120 characters
            - Strip and normalise spaces

        This is only for testing the experiment logic, not for quality.
        """
        text = prompt.strip().replace("\n", " ")
        if len(text) > 120:
            text = text[:120] + "..."
        return text


# ----------------------------
# Metrics
# ----------------------------

def compute_metrics(examples: List[Example]) -> Dict[str, Any]:
    """
    Compute simple metrics on a list of examples with predictions.
    """
    total = len(examples)
    if total == 0:
        return {
            "total": 0,
            "exact_match": 0,
            "exact_match_rate": 0.0,
            "avg_length_diff": 0.0,
        }

    exact = 0
    length_diffs: List[int] = []

    for ex in examples:
        if ex.prediction is None:
            continue
        # Clean text before comparison
        cleaned_prediction = clean_text(ex.prediction)
        cleaned_expected = clean_text(ex.expected)
        if cleaned_prediction == cleaned_expected:
            exact += 1
        length_diffs.append(abs(len(cleaned_prediction) - len(cleaned_expected)))

    avg_len_diff = statistics.mean(length_diffs) if length_diffs else 0.0
    exact_rate = exact / total if total > 0 else 0.0

    return {
        "total": total,
        "exact_match": exact,
        "exact_match_rate": exact_rate,
        "avg_length_diff": avg_len_diff,
    }


# ----------------------------
# Experiment runner
# ----------------------------

def run_experiment(dataset_path: str, model_name: str = "stub") -> Dict[str, Any]:
    """
    Run a tiny text experiment:

        - Load dataset from dataset_path
        - Create a client for the given model_name
        - Generate predictions for each example
        - Compute simple metrics
        - Return a metrics dict
    """
    print(f"[info] loading dataset from {dataset_path}")
    examples = load_dataset(dataset_path)
    client = LLMClient(model_name=model_name)

    print(f"[info] running model '{model_name}' on {len(examples)} examples")

    for ex in examples:
        ex.prediction = client.generate(ex.prompt)

    metrics = compute_metrics(examples)
    return metrics


# ----------------------------
# Entry point
# ----------------------------

if __name__ == "__main__":
    dataset = "data/sample.jsonl"
    report = run_experiment(dataset_path=dataset, model_name="stub")

    print("\nExperiment report")
    print("-----------------")
    for key, value in report.items():
        print(f"{key}: {value}")        
      
