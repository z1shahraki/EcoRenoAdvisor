import pandas as pd
import pytest
from pathlib import Path
from typing import List

import agent.tools as agent_tools


VOC_MAP = {
    "zero": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


@pytest.fixture(scope="session")
def materials_sample_path() -> Path:
    """Return the path to the sample materials CSV."""
    return Path(__file__).parent / "data" / "materials_sample.csv"


@pytest.fixture(scope="session")
def materials_df(materials_sample_path: Path) -> pd.DataFrame:
    """Load the sample materials CSV and add helper columns."""
    df = pd.read_csv(materials_sample_path)
    df["voc_level_num"] = (
        df["voc_level"].str.lower().map(VOC_MAP).fillna(df["voc_level"].str.lower())
    )
    return df


@pytest.fixture(autouse=False)
def patch_materials(monkeypatch: pytest.MonkeyPatch, materials_df: pd.DataFrame):
    """
    Monkeypatch MaterialsFilter to use the in-memory dataframe instead of parquet.
    """

    def _materials_property(self) -> pd.DataFrame:  # pragma: no cover - helper
        return materials_df.copy()

    monkeypatch.setattr(
        agent_tools.MaterialsFilter,
        "materials",
        property(_materials_property),
        raising=False,
    )
    return materials_df


@pytest.fixture(scope="session")
def sample_sentences() -> List[str]:
    """Provide a few simple sentences for embedding tests."""
    return [
        "low VOC water based paint for kids bedrooms",
        "ceramic tiles for kitchens",
        "wool carpet for living rooms",
    ]

