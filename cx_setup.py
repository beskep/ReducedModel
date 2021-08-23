from pathlib import Path

from cx_Freeze import Executable
from cx_Freeze import setup
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtQml

base = None
# if sys.platform == 'win32':
#   base = 'Win32GUI'

include_files = ['src', 'data']
includes = [
    'loguru',
    'rich',
    'rich.console',
    'rich.logging',
    'rich.progress',
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtQml',
    'PyQt5.QtQuick',
    'PyQt5.QtWidgets',
    'matplotlib',
    'seaborn',
    'numpy',
    'scipy',
    'scipy.spatial.transform._rotation_groups',
    'control',
    'slycot',
]
excludes = ['PySide2', 'email', 'pdb', 'pydoc', 'tkinter']
zip_include_packages = []

try:
  import mkl

  includes.append('mkl')

  libbin = Path(mkl.__path__[0]).parents[2].joinpath('Library/bin')
  for dll in ['mkl_intel_thread']:
    dll_path = libbin.joinpath(dll + '.dll')
    if not dll_path.exists():
      raise FileNotFoundError(dll_path)

    include_files.append(dll_path.as_posix())
except ImportError:
  pass

options = {
    'build_exe': {
        'include_files': include_files,
        'includes': includes,
        'zip_include_packages': zip_include_packages,
        'excludes': excludes,
    }
}

executables = [
    Executable('run.py', base=base),
]

setup(name='app',
      version='0.1',
      description='description',
      options=options,
      executables=executables)
