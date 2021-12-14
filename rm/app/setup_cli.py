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
  excludes = ['email', 'mypy', 'pdb', 'pydoc', 'PyQt5', 'tkinter']
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
