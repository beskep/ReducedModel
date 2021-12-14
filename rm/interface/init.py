import os
from pathlib import Path
import sys

from loguru import logger

from rm.utils import DIR
from rm.utils import IS_FROZEN


def init_project():
  # pylint: disable=import-outside-toplevel

  if not IS_FROZEN and 'PySide2' in sys.modules:
    import PySide2  # type: ignore

    pyside_dir = Path(PySide2.__file__).parent
    plugins_dir = pyside_dir.joinpath('plugins')
    qml_dir = pyside_dir.joinpath('qml')

    os.environ['QT_PLUGIN_PATH'] = plugins_dir.as_posix()
    os.environ['QML2_IMPORT_PATH'] = qml_dir.as_posix()

  import matplotlib as mpl
  import matplotlib.font_manager as fm

  mpl.use('Qt5Agg')

  font_name = 'Source Han Sans KR'
  font_path = DIR.RESOURCE.joinpath('font/SourceHanSansKR-Regular.otf')
  if not font_path.exists():
    logger.warning('Font not found: {}', font_path)
    return

  fe = fm.FontEntry(fname=font_path.as_posix(), name=font_name)
  fm.fontManager.ttflist.insert(0, fe)

  try:
    import seaborn as sns
  except ImportError:
    return

  mpl.rcParams['font.family'] = font_name
  mpl.rcParams['axes.unicode_minus'] = False

  sns.set_theme(context='notebook',
                style='whitegrid',
                font=font_name,
                rc={
                    'axes.edgecolor': '0.2',
                    'grid.color': '0.8'
                })
