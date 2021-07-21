import context
import utils

import numpy as np
import pytest

from reduced_model.matrix_reader import (MatrixReader, SystemMatricesReader,
                                         read_matrix)

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])

path = utils.ROOT_DIR.joinpath('test/test_matrix.txt')
DATA_DIR = utils.ROOT_DIR.joinpath('data')


def write_test_matrix():
  with open(path, 'w') as f:
    f.writelines(['*\n', '*\n', '*\n'])

    for r, c, d in np.vstack((row, col, data)).T:
      f.write(f'{r+1},0,{c+1},0,{d}\n')


def test_count_skip_rows():
  skip_row = MatrixReader._count_skip_rows(path)
  assert skip_row == 3


def test_read_matrix():
  mtx = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 6]])

  mtx_read1 = MatrixReader(path).read_matrix()
  mtx_read2 = read_matrix(path)

  assert mtx == pytest.approx(mtx_read1.toarray())
  assert mtx == pytest.approx(mtx_read2.toarray())


def test_read_symmetric_matrix():
  path = DATA_DIR.joinpath('test_case_simple/simple_modelingTHERM1_STIF1.mtx')
  mtx = MatrixReader(path, symmetric=True).read_matrix().toarray()

  assert mtx == pytest.approx(mtx.T)


def test_system_matrices_read():
  matrices = SystemMatricesReader(damping=path,
                                  stiffness=None,
                                  internal_load=None,
                                  external_load=None)
  mtx_read = matrices.damping_matrix
  mtx = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 6]])

  assert mtx == pytest.approx(mtx_read.toarray())


def test_system_matrices_error():
  # pylint: disable=pointless-statement
  matrices = SystemMatricesReader(damping=path,
                                  stiffness='./nonexist',
                                  internal_load=None,
                                  external_load=None)

  with pytest.raises(FileNotFoundError):
    matrices.stiffness_matrix

  with pytest.raises(FileNotFoundError):
    matrices.internal_load_matrix


def test_find_max_node():

  def _max_node_assert(file, value):
    max_node = SystemMatricesReader._find_max_node(file)
    assert max_node == value

  _max_node_assert(DATA_DIR.joinpath('test_case/C.txt'), 3270)
  _max_node_assert(DATA_DIR.joinpath('test_case/specific1.txt'), 2371)
  _max_node_assert(
      DATA_DIR.joinpath('test_case_simple/C_HcombTHERM1_DMPV1.mtx'), 168)


if __name__ == '__main__':
  pytest.main(['-vv', '-k', 'test_matrix_reader'])
