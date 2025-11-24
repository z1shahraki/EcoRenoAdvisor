"""
Extract text from PDF documents.

This script:
- Scans data/raw/ for PDF files
- Extracts text from each page
- Saves as JSONL (one JSON object per line)
"""

from pypdf import PdfReader
import json
import glob
from pathlib import Path


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract all text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text as a single string
    """
    reader = PdfReader(pdf_path)
    text_parts = []
    
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    
    return " ".join(text_parts)


def extract_all_docs(input_dir: str, output_path: str) -> None:
    """
    Extract text from all PDFs in input_dir and save as JSONL.
    
    Args:
        input_dir: Directory containing PDF files
        output_path: Path to save JSONL output
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    pdf_files = glob.glob(f"{input_dir}/*.pdf")
    
    if not pdf_files:
        print(f"WARNING: No PDF files found in {input_dir}")
        return
    
    output = []
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path}")
        try:
            text = extract_pdf_text(pdf_path)
            output.append({
                "source": pdf_path,
                "text": text
            })
            print(f"  Extracted {len(text)} characters")
        except Exception as e:
            print(f"  ERROR: Error processing {pdf_path}: {e}")
    
    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(output)} documents to {output_path}")


if __name__ == "__main__":
    input_dir = "data/raw"
    output_jsonl = "data/clean/docs_text.jsonl"
    
    extract_all_docs(input_dir, output_jsonl)

