import context
import utils

import numpy as np

from reduced_model.matrix_reader import read_matrix
from reduced_model.state_space import SystemH

data_dir = utils.ROOT_DIR.joinpath('data/optimize')
path_C = data_dir.joinpath('C_o.txt')
path_Ns = [data_dir.joinpath('specific1_node.txt')]


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


def make_system_h(Ti=20.0, Te=5.0):
  H = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)

  K = [Files.get_path(Matrices.K, x[0], x[1]) for x in H]
  Li = [Files.get_path(Matrices.Li, x[0], x[1]) for x in H]
  Le = [Files.get_path(Matrices.Le, x[0], x[1]) for x in H]

  system_h = SystemH.from_files(H=H,
                                C=path_C,
                                K=K,
                                Li=Li,
                                Le=Le,
                                Ti=Ti,
                                Te=Te,
                                Ns=path_Ns)

  return system_h
