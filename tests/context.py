import sys
from pathlib import Path

TEST_DIR = Path(__file__).parent
ROOT_DIR = TEST_DIR.parent
SRC_DIR = ROOT_DIR.joinpath('src')

_SRC_DIR = SRC_DIR.as_posix()

if _SRC_DIR not in sys.path:
  sys.path.append(_SRC_DIR)
