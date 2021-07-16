import argparse
from pathlib import Path

import context
import utils

import numpy as np
import scipy.io
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True)
parser.add_argument('-d', '--directory', required=True)


def read_and_save_mat(path, save_dir):
  path = Path(path).resolve()
  if not path.exists():
    raise FileNotFoundError(path)

  save_dir = Path(save_dir).resolve()
  if not save_dir.exists():
    raise FileNotFoundError(save_dir)

  mat: dict = scipy.io.loadmat(path.as_posix())

  for key, value in mat.items():
    logger.info(
        'key: {} | value: `{}...`',
        key,
        str(value)[:10].replace('\n', ''),
    )

    if isinstance(value, np.ndarray):
      save_path = save_dir.joinpath(f'{key}.txt')

      try:
        np.savetxt(save_path.as_posix(), value, delimiter=',')
      except (ValueError, TypeError):
        logger.error('{} is not numeric array', key)

        with save_path.open('w', encoding='utf-8') as f:
          f.write(str(value))


if __name__ == '__main__':
  utils.set_logger(level='DEBUG')

  args = parser.parse_args()

  logger.debug(args)

  read_and_save_mat(path=args.file, save_dir=args.directory)
