"""
Utility functions for text processing
"""


def clean_text(text: str) -> str:
    """
    Remove extra spaces and convert text to lowercase.
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text with normalized spaces and lowercase
    """
    # Convert to lowercase
    cleaned = text.lower()
    # Remove extra spaces by splitting and rejoining
    cleaned = " ".join(cleaned.split())
    return cleaned

