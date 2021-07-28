"""
ABAQUS에서 추출한 행렬 파일 해석

C = Damping = Capacitance
K = Stiffness = Conductance (symmetric)
L = Load = Solicitation
"""

import os
from collections import deque
from functools import cached_property
from pathlib import Path
from typing import Iterable, Optional, Tuple

from utils import StrPath

import numpy as np
from scipy.sparse import csc_matrix


def _get_node_number(line: str):
  loc = line.find(MatrixReader.DELIMITER)
  if loc == -1:
    node = np.nan
  else:
    try:
      node = int(float(line[:loc]))
    except ValueError:
      node = np.nan

  return node


class MatrixReader:
  COMMENT_PREFIX = '*'
  DELIMITER = ','

  def __init__(self,
               path: Optional[StrPath],
               shape: Optional[Tuple[int, int]] = None,
               symmetric=False) -> None:
    self._path = Path(path) if path else None
    self._shape = shape
    self._symmetric = symmetric

  @property
  def path(self):
    return self._path

  @path.setter
  def path(self, value: StrPath):
    self._path = Path(value)

  @classmethod
  def _count_skip_rows(cls, path):
    count = 0

    with open(path, 'r') as f:
      while True:
        line = f.readline()
        if not (line and line.startswith(cls.COMMENT_PREFIX)):
          break

        count += 1

    return count

  @staticmethod
  def _symmetric_index(value: np.ndarray, row: np.ndarray, col: np.ndarray):
    assert value.ndim == 1
    assert value.shape == row.shape
    assert value.shape == col.shape

    diagonal_mask = row == col
    diag_indices = np.where(diagonal_mask)
    nondiag_indices = np.where(np.logical_not(diagonal_mask))

    symm_value = np.concatenate([
        value[diag_indices],
        value[nondiag_indices],
        value[nondiag_indices],
    ])
    symm_row = np.concatenate([
        row[diag_indices],
        row[nondiag_indices],
        col[nondiag_indices],
    ])
    symm_col = np.concatenate([
        col[diag_indices],
        col[nondiag_indices],
        row[nondiag_indices],
    ])

    return symm_value, symm_row, symm_col

  def read_matrix(self) -> csc_matrix:
    if not (self._path and self._path.exists()):
      raise FileNotFoundError(self._path)

    # sparse matrix 형태 파일 읽기
    skip_row = self._count_skip_rows(self._path)
    raw: np.ndarray = np.loadtxt(fname=self._path.as_posix(),
                                 delimiter=self.DELIMITER,
                                 skiprows=skip_row)

    if raw.shape[1] > 2:
      # raw에서 동일한 숫자만 있는 column 제거
      identical_cols = np.all(raw == raw[0, :], axis=0)
      assert np.sum(identical_cols) != raw.shape[1]
      raw = raw[:, np.logical_not(identical_cols)]

    # 각 column의 값을 value, row, col에 지정
    value = raw[:, -1]
    row = (raw[:, 0] - 1).astype(int)  # matlab matrix index는 1부터 시작
    if raw.shape[1] == 3:
      col = (raw[:, 1] - 1).astype(int)
    elif raw.shape[1] == 2:
      # load matrix의 경우 한 줄짜리 column (shape==(rows, 1))
      col = np.repeat(0, row.shape[0])
    else:
      raise ValueError

    # stiffness matrix (symmetric임)인 경우, lower/upper triangle 복사
    if self._symmetric:
      value, row, col = self._symmetric_index(value, row, col)

    # matrix shape 지정
    if self._shape is None:
      shape = tuple(int(np.max(x)) + 1 for x in [row, col])
    else:
      shape = self._shape

    mat = csc_matrix((value, (row, col)), shape=shape)

    return mat


def read_matrix(path: StrPath,
                shape: Optional[Tuple[int, int]] = None,
                symmetric=False) -> csc_matrix:
  return MatrixReader(path=path, shape=shape, symmetric=symmetric).read_matrix()


class MatricesReader:
  """
  개별 matrix 파일에 최대 노드 번호가 누락되었을 경우에 대비해서,
  모든 파일로부터 max_node를 먼저 읽고 각 matrix의 shape을 결정
  """

  def __init__(self, files: Iterable[StrPath] = None, max_node=0) -> None:
    if files is None and max_node is None:
      raise ValueError('files나 max_node 중 하나는 지정해야 합니다.')

    if files is None:
      files = []

    # max_node, 지정한 files의 최대 노드 중 최댓값을 self._max_node로 설정
    max_nodes = [self._find_max_node(x) for x in files]
    self._max_node = max([max_node] + max_nodes)

    if self._max_node <= 0:
      raise ValueError('max_node <= 0')

  @property
  def max_node(self):
    return self._max_node

  @staticmethod
  def _find_max_node(path: StrPath):
    """
    mtx파일에서 최대 노드 번호를 찾아 반환
    대상 파일이 존재하지 않는 경우 -1 반환
    """
    if not (path and os.path.exists(path)):
      raise FileNotFoundError(path)

    lines = deque(maxlen=2)

    try:
      with open(path, 'r') as f:
        while True:
          line = f.readline()
          if line:
            lines.append(line)
          else:
            break

      max_node = np.nanmax([_get_node_number(line) for line in lines])
      if np.isnan(max_node):
        raise ValueError

    except ValueError as e:
      raise ValueError('파일 형식 오류: {}'.format(path)) from e

    return max_node

  def read_matrix(self,
                  path: StrPath,
                  square: bool,
                  symmetric=False) -> csc_matrix:
    if not square and symmetric:
      raise ValueError

    if square:
      shape = (self.max_node, self.max_node)
    else:
      shape = (self.max_node, 1)

    matrix = read_matrix(path=path, shape=shape, symmetric=symmetric)

    return matrix


class SystemMatricesReader(MatricesReader):

  def __init__(self,
               damping: StrPath,
               stiffness: StrPath,
               internal_load: StrPath,
               external_load: StrPath,
               max_node=0) -> None:
    self._damping = damping
    self._stiffness = stiffness
    self._internal_load = internal_load
    self._external_load = external_load

    super().__init__(files=[damping, stiffness, internal_load, external_load],
                     max_node=max_node)

  @cached_property
  def damping_matrix(self):
    return self.read_matrix(path=self._damping, square=True, symmetric=False)

  @cached_property
  def stiffness_matrix(self):
    return self.read_matrix(path=self._stiffness, square=True, symmetric=True)

  @cached_property
  def internal_load_matrix(self):
    return self.read_matrix(path=self._internal_load,
                            square=False,
                            symmetric=False)

  @cached_property
  def external_load_matrix(self):
    return self.read_matrix(path=self._external_load,
                            square=False,
                            symmetric=False)

  def load_matrices(self):
    return self.internal_load_matrix, self.external_load_matrix


if __name__ == '__main__':
  root_dir = Path(__file__).parents[1]
  matrix = SystemMatricesReader(
      damping=root_dir.joinpath('test/C_HcombTHERM1_DMPV1.mtx'),
      stiffness=root_dir.joinpath('test/K_HcombTHERM1_STIF1.mtx'),
      internal_load=root_dir.joinpath('test/Lin_HinTHERM1_LOAD1.mtx'),
      external_load=root_dir.joinpath('test/Lout_HoutTHERM1_LOAD1.mtx'))

  end = 3
  print(matrix.damping_matrix[0:end, 0:end])
  print(matrix.internal_load_matrix[0:end, 0:end])
  print(matrix.stiffness_matrix[0:end, 0:end])
