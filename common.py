"""Shared imports, paths and constants for every stage.

Usage in any script / notebook:
    from common import *
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- paths ------------------------------------------------------------------
RAW = Path("data/car-hacking-dataset/Car-Hacking")   # original HCRL files
PROC = Path("data/processed")                        # Stage 1 output
FEAT = Path("data/features")                         # Stage 2 output
FIG = Path("reports/figures")
for _p in (PROC, FEAT, FIG):
    _p.mkdir(parents=True, exist_ok=True)

# ---- dataset ----------------------------------------------------------------
# name -> file name. EDIT these if your files are named differently.
FILES = {
    "DoS": "DoS_dataset.csv",
    "Fuzzy": "Fuzzy_dataset.csv",
    "Gear": "gear_dataset.csv",
    "RPM": "RPM_dataset.csv",
    "Normal": "normal_run_data.txt",
}
ATTACKS = ["DoS", "Fuzzy", "Gear", "RPM"]

BYTE_COLS = [f"b{i}" for i in range(8)]
META_COLS = ["timestamp", "can_id", "dlc", *BYTE_COLS, "label", "attack_type"]
HEX = {f"{i:02x}": i for i in range(256)} | {f"{i:02X}": i for i in range(256)}

GAP = 10  # seconds; a larger timestamp gap = a new recording

__all__ = [
    "json", "re", "Path", "plt", "np", "pd",
    "RAW", "PROC", "FEAT", "FIG",
    "FILES", "ATTACKS", "BYTE_COLS", "META_COLS", "HEX", "GAP",
]
