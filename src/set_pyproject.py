import json
from pathlib import Path

ROOT_DIR = Path(__file__).parents[1]
pyproject_path = ROOT_DIR.joinpath('main.pyproject')
assert pyproject_path.exists()

exts = ['.py', '.qml']

files = []
for ext in exts:
  files_ = (set(ROOT_DIR.rglob(f'*{ext}')) -
            set(ROOT_DIR.rglob(f'dist/**/*{ext}')) -
            set(ROOT_DIR.rglob(f'build/**/*{ext}')))
  files_ = [f.relative_to(ROOT_DIR).as_posix() for f in files_]
  files.extend(files_)

files.sort()

with open(pyproject_path, 'w') as f:
  json.dump({'files': files}, fp=f, indent=2)
