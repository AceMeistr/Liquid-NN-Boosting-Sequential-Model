"""Root conftest — adds src/ to sys.path so all test modules can import modelmk1."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
