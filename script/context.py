import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
SRC_DIR = ROOT_DIR.joinpath('src')

_SRC_DIR = SRC_DIR.as_posix()

if _SRC_DIR not in sys.path:
  sys.path.append(_SRC_DIR)
