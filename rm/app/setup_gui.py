from pathlib import Path
import sys

from cx_Freeze import Executable
from cx_Freeze import setup

from rm import utils

if __name__ == '__main__':
  include_files = ['sample', 'resource']
  includes = [
      'control',
      'loguru',
      'matplotlib',
      'numpy',
      'rich.console',
      'rich.logging',
      'rich.progress',
      'rich',
      'scipy.spatial.transform._rotation_groups',
      'scipy',
      'seaborn.cm',
      'seaborn',
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

  excludes = ['mypy', 'pdb', 'resource', 'tkinter']
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
      Executable('rm/app/cli.py', target_name='CLI'),
      Executable('rm/app/gui.py', target_name='GUI'),
  ]

  setup(name='app',
        version='0.1',
        description='description',
        options=options,
        executables=executables)
