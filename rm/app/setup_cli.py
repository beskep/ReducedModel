from pathlib import Path
import sys

from cx_Freeze import Executable
from cx_Freeze import setup

if __name__ == '__main__':
  include_files = ['sample']
  includes = [
      'loguru',
      'rich',
      'rich.console',
      'rich.logging',
      'rich.progress',
      'numpy',
      'scipy',
      'scipy.spatial.transform._rotation_groups',
      'control',
      'slycot',
  ]

  try:
    import mkl  # type: ignore
  except ImportError:
    pass
  else:
    includes.append('mkl')

    bindir = Path(sys.base_prefix).joinpath('Library/bin')
    for p in ('mkl_intel', 'mkl_core', 'mkl_def', 'libiomp'):
      bins = bindir.glob(f'{p}*.dll')
      include_files.extend([x.as_posix() for x in bins])

  excludes = ['email', 'mypy', 'pdb', 'pydoc', 'PyQt5', 'resource', 'tkinter']
  zip_include_packages = []

  options = {
      'build_exe': {
          'include_files': include_files,
          'includes': includes,
          'zip_include_packages': zip_include_packages,
          'excludes': excludes,
      }
  }

  executables = [
      Executable('rm/app/cli.py'),
  ]

  setup(name='app',
        version='0.1',
        description='description',
        options=options,
        executables=executables)
