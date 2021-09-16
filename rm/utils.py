from os import PathLike
from pathlib import Path
import sys
from typing import Union

from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

IS_FROZEN = getattr(sys, 'frozen', False)


class DIR:
  if getattr(sys, 'frozen', False):
    ROOT = Path(sys.executable).parent.resolve()
  else:
    ROOT = Path(__file__).parents[1].resolve()

  SRC = ROOT.joinpath('rm')
  RESOURCE = ROOT.joinpath('resource')


console = Console()
StrPath = Union[str, PathLike[str]]


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
