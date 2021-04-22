import utils

import scipy.io
import numpy as np

if __name__ == '__main__':
  mat: dict = scipy.io.loadmat(
      utils.ROOT_DIR.joinpath('data/Reduction_model_matfile.mat'))

  save_dir = utils.ROOT_DIR.joinpath('res')
  if not save_dir.exists():
    save_dir.mkdir()

  for key, value in mat.items():
    print(f'key: {key} | value: {value}')
    if isinstance(value, np.ndarray):
      save_path = save_dir.joinpath(f'{key}.txt')
      np.savetxt(save_path.as_posix(), value, delimiter=',')
