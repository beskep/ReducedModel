from cx_Freeze import Executable
from cx_Freeze import setup

if __name__ == '__main__':
  include_files = ['data', 'resource']
  includes = [
      'loguru',
      'rich',
      'rich.console',
      'rich.logging',
      'rich.progress',
      'matplotlib',
      'seaborn',
      'seaborn.cm',
      'numpy',
      'scipy',
      'scipy.spatial.transform._rotation_groups',
      'control',
      'slycot',
  ]
  excludes = ['mypy', 'pdb', 'tkinter']
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
