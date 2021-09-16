import os
from pathlib import Path
import sys

from rm.utils import IS_FROZEN


def init_project():
  # pylint: disable=import-outside-toplevel

  if not IS_FROZEN and 'PySide2' in sys.modules:
    import PySide2

    pyside_dir = Path(PySide2.__file__).parent
    plugins_dir = pyside_dir.joinpath('plugins')
    qml_dir = pyside_dir.joinpath('qml')

    os.environ['QT_PLUGIN_PATH'] = plugins_dir.as_posix()
    os.environ['QML2_IMPORT_PATH'] = qml_dir.as_posix()

  import matplotlib as mpl

  mpl.use('Qt5Agg')
