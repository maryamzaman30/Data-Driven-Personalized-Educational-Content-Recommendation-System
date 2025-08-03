import os
import sys
from pathlib import Path

# Get the absolute path of the project root (one level up from tests/)
project_root = Path(__file__).parent.parent.absolute()

# Add the project root and src directory to the Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

# Optional: Print the paths for debugging
print("Python path:", sys.path, file=sys.stderr)
