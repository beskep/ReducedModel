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

options = {
    'build_exe': {
        'include_files': include_files,
        'includes': includes,
        'zip_include_packages': zip_include_packages,
        'excludes': excludes,
    }
}

executables = [
    Executable('rm/app/gui.py'),
]

setup(name='app',
      version='0.1',
      description='description',
      options=options,
      executables=executables)