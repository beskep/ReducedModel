import context
import utils

from reduced_model.matrix_reader import read_matrix

data_dir = utils.ROOT_DIR.joinpath('data/optimize')


class Matrices:
  K = 'K'
  Li = 'Lin'
  Le = 'Lout'

  all_ = [K, Li, Le]


class Files:
  _matrices = {
      (1, 1): '_o_1',  # (h_interior, h_exterior): matrix
      (2, 2): '_o_2',
      (1, 2): '_o_3',
      (2, 1): '_o_4',
  }

  @classmethod
  def matrices(cls, m):
    if m not in Matrices.all_:
      raise ValueError

    return {key: m + value for key, value in cls._matrices.items()}

  @classmethod
  def get_path(cls, m: str, hi: int, he: int):
    matrices = cls.matrices(m)
    name = matrices[(int(hi), int(he))]
    path = data_dir.joinpath(f'{name}.txt')

    return path

  @classmethod
  def read_matrix(cls, m: str, hi: int, he: int, symm):
    path = cls.get_path(m, hi, he)
    matrix = read_matrix(path=path, symmetric=symm)

    return matrix
