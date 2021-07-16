import sys
from pathlib import Path
from typing import Union

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

_SRC_DIR = SRC_DIR.as_posix()
if _SRC_DIR not in sys.path:
  sys.path.append(_SRC_DIR)

console = Console()


def set_logger(level: Union[int, str, None] = None):
  logger.remove()

  if level is None:
    if any('debug' in x.lower() for x in sys.argv):
      level = 'DEBUG'
    else:
      level = 'INFO'

  rich_handler = RichHandler(console=console, log_time_format='[%y-%m-%d %X]')
  logger.add(rich_handler, level=level, format='{message}', enqueue=True)
  logger.add('rm.log',
             level='DEBUG',
             rotation='1 week',
             retention='1 month',
             encoding='UTF-8-SIG',
             enqueue=True)
