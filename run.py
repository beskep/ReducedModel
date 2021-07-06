import os
import sys

sys.path.insert(0, os.path.normpath('./src'))

# pylint: disable=wrong-import-position
from main import main

main()
