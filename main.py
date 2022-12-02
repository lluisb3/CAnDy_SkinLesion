# Main .py file to run the project
from pathlib import Path
thispath = Path(__file__).resolve()
import sys


print(thispath.parent)
sys.path.insert(0, str(thispath.parent))
