"""
Clean and normalize materials CSV data.

This script:
- Reads raw materials CSV
- Normalizes price fields (removes $, spaces)
- Maps VOC levels to numeric values
- Saves cleaned data as Parquet
"""

import pandas as pd
from pathlib import Path


def clean_materials(input_path: str, output_path: str) -> None:
    """
    Clean materials CSV and save as Parquet.
    
    Args:
        input_path: Path to raw materials.csv
        output_path: Path to save cleaned materials.parquet
    """
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read CSV
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {input_path}: {e}")
    
    # Clean price_per_m2: remove $, spaces, commas
    if 'price_per_m2' in df.columns:
        df['price_per_m2'] = df['price_per_m2'].astype(str).replace(r'[$, ]', '', regex=True)
        df['price_per_m2'] = pd.to_numeric(df['price_per_m2'], errors='coerce')
    
    # Map VOC levels to numeric
    if 'voc_level' in df.columns:
        mapping = {"low": 1, "medium": 2, "high": 3, "zero": 0}
        df['voc_level_num'] = df['voc_level'].str.lower().map(mapping)
        # Fill NaN with 2 (medium) as default
        df['voc_level_num'] = df['voc_level_num'].fillna(2)
    
    # Ensure eco_score is numeric
    if 'eco_score' in df.columns:
        df['eco_score'] = pd.to_numeric(df['eco_score'], errors='coerce')
        df['eco_score'] = df['eco_score'].fillna(0.5)  # Default to 0.5 if missing
    
    # Save cleaned data
    try:
        df.to_parquet(output_path, index=False)
        print(f"Saved cleaned materials to {output_path}")
        print(f"  Total materials: {len(df)}")
    except Exception as e:
        raise IOError(f"Error saving cleaned data to {output_path}: {e}")


if __name__ == "__main__":
    input_csv = "data/raw/materials.csv"
    output_parquet = "data/clean/materials.parquet"
    
    clean_materials(input_csv, output_parquet)

