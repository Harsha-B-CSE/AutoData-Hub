import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pytest
import pandas as pd
from core.dataset_manager import DatasetManager

def test_scenario_c_tier1_immutability():
    manager = DatasetManager()
    df_raw = pd.DataFrame({"col1":[1,2,3], "col2": ["A", "B", "C"]})
    manager.register_version("v0_raw", df_raw)
    df_retrieved = manager.get_version("v0_raw")
    assert id(df_raw) != id(df_retrieved)
    df_retrieved.loc[0, "col1"] = 99
    assert manager.get_version("v0_raw")["col1"].iloc[0] == 1
