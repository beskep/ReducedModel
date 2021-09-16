from pathlib import Path
import sys

from cx_Freeze import Executable
from cx_Freeze import setup

include_files = ['data', 'resource']
includes = [
    'loguru',
    'rich',
    'rich.console',
    'rich.logging',
    'rich.progress',
    'matplotlib',
    'seaborn',
    'numpy',
    'scipy',
    'scipy.spatial.transform._rotation_groups',
    'control',
    'slycot',
]
excludes = ['email', 'pdb', 'pydoc', 'tkinter']
zip_include_packages = []

try:
  import mkl
except ImportError:
  pass
else:
  includes.append('mkl')

  libbin = Path(sys.base_prefix).joinpath('Library/bin')
  for dll in ['mkl_intel_thread']:
    dll_path = libbin.joinpath(dll + '.dll')
    if not dll_path.exists():
      raise FileNotFoundError(dll_path)

    include_files.append(dll_path.as_posix())

options = {
    'build_exe': {
        'include_files': include_files,
        'includes': includes,
        'zip_include_packages': zip_include_packages,
        'excludes': excludes,
    }
}

executables = [
    Executable('run.py'),
]

setup(name='app',
      version='0.1',
      description='description',
      options=options,
      executables=executables)
