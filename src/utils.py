import sys
from pathlib import Path

from loguru import logger

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent

_SRC_DIR = SRC_DIR.as_posix()

if _SRC_DIR not in sys.path:
  sys.path.append(_SRC_DIR)


def set_logger():
  fmt = ('<green>{time:YYYY-MM-DD HH:mm:ss}</green> | '
         '<level>{level: <8}</level> | '
         '<cyan>{file}</cyan>:<cyan>{line}</cyan> - '
         '<level>{message}</level>')

  logger.remove()
  logger.add(sys.stdout, level='INFO', format=fmt, enqueue=True)
  logger.add('.log', rotation='1 MB', encoding='utf-8-sig', enqueue=True)
