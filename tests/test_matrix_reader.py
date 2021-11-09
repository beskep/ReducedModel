import numpy as np
import pytest

from rm import utils
import rm.reduced_model.matrix_reader as mr

row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])

path = utils.DIR.ROOT.joinpath('tests/test_matrix.txt')
DATA_DIR = utils.DIR.ROOT.joinpath('data')


def write_test_matrix():
  with open(path, 'w') as f:
    f.writelines(['*\n', '*\n', '*\n'])

    for r, c, d in np.vstack((row, col, data)).T:
      f.write(f'{r+1},0,{c+1},0,{d}\n')


def test_count_skip_rows():
  skip_row = mr.MatrixReader._count_skip_rows(path)
  assert skip_row == 3


def test_read_matrix():
  mtx = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 6]])

  mtx_read1 = mr.MatrixReader(path).read_matrix()
  mtx_read2 = mr.read_matrix(path)

  assert mtx == pytest.approx(mtx_read1.toarray())
  assert mtx == pytest.approx(mtx_read2.toarray())


def test_read_symmetric_matrix():
  path = DATA_DIR.joinpath('test_case_simple/simple_modelingTHERM1_STIF1.mtx')
  mtx = mr.MatrixReader(path, symmetric=True).read_matrix().toarray()

  assert mtx == pytest.approx(mtx.T)


def test_matrices_reader_max_node():
  assert mr.MatricesReader(files=[path, path], max_node=-1).max_node == 3
  assert mr.MatricesReader(files=None, max_node=99).max_node == 99

  assert mr.MatricesReader(files=[path], max_node=99).max_node == 99
  assert mr.MatricesReader(files=[path, path], max_node=-7).max_node == 3

  with pytest.raises(ValueError):
    mr.MatricesReader(files=None, max_node=None)


def test_system_matrices_read():
  matrices = mr.SystemMatricesReader(damping=path,
                                     stiffness=path,
                                     internal_load=path,
                                     external_load=path)
  mtx_read = matrices.damping_matrix
  mtx = np.array([[1, 0, 2], [0, 0, 3], [4, 5, 6]])

  assert mtx == pytest.approx(mtx_read.toarray())


def test_system_matrices_error():
  # pylint: disable=pointless-statement
  with pytest.raises(FileNotFoundError):
    mr.SystemMatricesReader(damping=path,
                            stiffness='./nonexist',
                            internal_load=None,
                            external_load=None)


def test_find_max_node():

  def _max_node_assert(file, value):
    max_node = mr.SystemMatricesReader._find_max_node(file)
    assert max_node == value

  _max_node_assert(DATA_DIR.joinpath('test_case/C.txt'), 3270)
  _max_node_assert(DATA_DIR.joinpath('test_case/specific1.txt'), 2371)
  _max_node_assert(
      DATA_DIR.joinpath('test_case_simple/C_HcombTHERM1_DMPV1.mtx'), 168)
