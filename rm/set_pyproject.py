import json
from pathlib import Path

if __name__ == '__main__':
  ROOT_DIR = Path(__file__).parents[1]
  pyproject_path = ROOT_DIR.joinpath('main.pyproject')
  assert pyproject_path.exists()

  exts = ['.py', '.qml']

  files = []
  for ext in exts:
    fs = (set(ROOT_DIR.rglob(f'*{ext}')) -
          set(ROOT_DIR.rglob(f'dist/**/*{ext}')) -
          set(ROOT_DIR.rglob(f'build/**/*{ext}')))
    fl = [f.relative_to(ROOT_DIR).as_posix() for f in fs]
    files.extend(fl)

  files.sort()

  with open(pyproject_path, 'w') as f:
    json.dump({'files': files}, fp=f, indent=2)
