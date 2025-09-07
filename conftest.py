# =========================
# File: conftest.py
# Purpose:
#   Ensure project root and src/ are on the Python path
#   so tests can import application modules directly.
# =========================

import sys
from pathlib import Path

# -------------------------
# 1. Determine Project Root
# -------------------------
# __file__ is the path to this file (tests/conftest.py)
# .parent → tests/
# .parent.parent → project root
project_root = Path(__file__).parent.parent.absolute()

# -------------------------
# 2. Add Project Directories to Python Path
# -------------------------
# Insert at the beginning so these paths are searched first
sys.path.insert(0, str(project_root))          # Project root
sys.path.insert(0, str(project_root / "src"))  # src/ directory

# -------------------------
# 3. Debug: Show sys.path during test runs
# -------------------------
print(f"[conftest] Python path set to: {sys.path}", file=sys.stderr)
