"""
대류 열전달계수에 따른 행렬 변환 테스트
"""
import context
import utils

import numpy as np
import pytest

import reduced_model.matrix_reader as mr
import reduced_model.system_matrix as sm

data_dir = utils.ROOT_DIR.joinpath('data/optimize')

matrices = {
    (1, 1): 'K_o_1',  # (h_interior, h_exterior): matrix
    (2, 2): 'K_o_2',
    (1, 2): 'K_o_3',
    (2, 1): 'K_o_4',
}


def get_path(hi, he):
  name = matrices[(hi, he)]

  return data_dir.joinpath(f'{name}.txt')


def read_K(hi, he):
  matrix = mr.read_matrix(path=get_path(hi=hi, he=he), is_symmetric=True)

  return matrix


def test_linear_hypothesis():
  """
  System matrix의 노드별 선형 결합 가설 검증

  가정:

  K(h_int, h_ext) = K0 + h_int * Kint + h_ext * Kext

  h_int: 접합부 내부 대류열전달계수 (scalar)
  h_ext: 접합부 외부 대류열전달계수 (scalar)
  K(h_int, h_ext): Conductance matrix
  """

  K11 = read_K(1, 1)
  K12 = read_K(1, 2)
  K21 = read_K(2, 1)
  K22 = read_K(2, 2)  # 검증용

  assert K11.shape == K12.shape
  assert K11.shape == K21.shape
  assert K11.shape == K22.shape

  Kint = K21 - K11
  Kext = K12 - K11
  K0 = K11 - Kint - Kext

  K22e = K0 + 2 * Kint + 2 * Kext
  assert K22e.shape == K22.shape

  # assert K22e.toarray() == pytest.approx(K22.toarray())  # 너무 느림
  assert np.allclose(K22.toarray(), K22e.toarray())


def test_linalg_solve():
  """
  `test_linear_hypothesis`를 행렬 식으로 풀이

  [h_int h_ext 1.0] [Kint Kext K0].T = K
  """

  K11 = read_K(1, 1)
  K12 = read_K(1, 2)
  K21 = read_K(2, 1)
  K22 = read_K(2, 2)  # 검증용

  A = np.array([
      [1, 1, 1],
      [1, 2, 1],
      [2, 1, 1],
  ], dtype=float)

  invA = np.linalg.inv(A)

  Kint = invA[0, 0] * K11 + invA[0, 1] * K12 + invA[0, 2] * K21
  Kext = invA[1, 0] * K11 + invA[1, 1] * K12 + invA[1, 2] * K21
  K0 = invA[2, 0] * K11 + invA[2, 1] * K12 + invA[2, 2] * K21

  K22e = K0 + 2 * Kint + 2 * Kext
  assert K22e.shape == K22.shape

  assert np.allclose(K22.toarray(), K22e.toarray())


# 이하 MatrixH 테스트


def test_inv_error():
  with pytest.raises(np.linalg.LinAlgError):
    sm.MatrixH(H=np.ones([3, 3]), Ms=[None, None, None])


def test_validation():
  K11 = read_K(1, 1)
  K12 = read_K(1, 2)
  K21 = read_K(2, 1)
  K22 = read_K(2, 2)

  H = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)
  mh = sm.MatrixH(H=H, Ms=[K11, K12, K21])

  K22e = mh.matrix(h_interior=2.0, h_exterior=2.0)

  assert np.allclose(K22.toarray(), K22e.toarray())


def test_from_files():
  K22 = read_K(2, 2)

  H = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)
  mh = sm.MatrixH.from_files(H=H,
                             files=[get_path(x[0], x[1]) for x in H],
                             is_square=True,
                             is_symmetric=True)

  K22e = mh.matrix(h_interior=2.0, h_exterior=2.0)

  assert np.allclose(K22.toarray(), K22e.toarray())


if __name__ == '__main__':
  pytest.main(['-vv', '-k', 'test_system_matrix'])
