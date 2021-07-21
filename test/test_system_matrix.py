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
    name = matrices[(hi, he)]
    path = data_dir.joinpath(f'{name}.txt')

    return path

  @classmethod
  def read_matrix(cls, m: str, hi: int, he: int, symm):
    path = cls.get_path(m, hi, he)
    matrix = mr.read_matrix(path=path, is_symmetric=symm)

    return matrix


class TestHypothesis:

  @pytest.mark.parametrize('m', Matrices.all_)
  def test_linear_hypothesis(self, m):
    """
    System matrix의 노드별 선형 결합 가설 검증

    가정:

    K(h_int, h_ext) = K0 + h_int * Kint + h_ext * Kext

    h_int: 접합부 내부 대류열전달계수 (scalar)
    h_ext: 접합부 외부 대류열전달계수 (scalar)
    K(h_int, h_ext): Conductance matrix

    K, Linterior, Lexterior 모두 마찬가지
    """
    symm = (m == 'K')
    M11 = Files.read_matrix(m, hi=1, he=1, symm=symm)
    M12 = Files.read_matrix(m, hi=1, he=2, symm=symm)
    M21 = Files.read_matrix(m, hi=2, he=1, symm=symm)
    M22 = Files.read_matrix(m, hi=2, he=2, symm=symm)  # 검증용

    assert M11.shape == M12.shape
    assert M11.shape == M21.shape
    assert M11.shape == M22.shape

    Mint = M21 - M11
    Mext = M12 - M11
    M0 = M11 - Mint - Mext

    M22e = M0 + 2 * Mint + 2 * Mext
    assert M22e.shape == M22.shape

    # assert K22e.toarray() == pytest.approx(K22.toarray())  # 너무 느림
    assert np.allclose(M22.toarray(), M22e.toarray())

  @pytest.mark.parametrize('m', Matrices.all_)
  def test_linalg_solve(self, m):
    """
    `test_linear_hypothesis`를 행렬 식으로 풀이

    [h_int h_ext 1.0] [Kint Kext K0].T = K
    """
    symm = (m == 'K')
    M11 = Files.read_matrix(m, hi=1, he=1, symm=symm)
    M12 = Files.read_matrix(m, hi=1, he=2, symm=symm)
    M21 = Files.read_matrix(m, hi=2, he=1, symm=symm)
    M22 = Files.read_matrix(m, hi=2, he=2, symm=symm)  # 검증용

    A = np.array([
        [1, 1, 1],
        [1, 2, 1],
        [2, 1, 1],
    ], dtype=float)

    invA = np.linalg.inv(A)

    Mint = invA[0, 0] * M11 + invA[0, 1] * M12 + invA[0, 2] * M21
    Mext = invA[1, 0] * M11 + invA[1, 1] * M12 + invA[1, 2] * M21
    M0 = invA[2, 0] * M11 + invA[2, 1] * M12 + invA[2, 2] * M21

    K22e = M0 + 2 * Mint + 2 * Mext
    assert K22e.shape == M22.shape

    assert np.allclose(M22.toarray(), K22e.toarray())


class TestMatrixH:

  def test_inv_error(self):
    with pytest.raises(np.linalg.LinAlgError):
      sm.MatrixH(H=np.ones([3, 3]), Ms=[None, None, None])

  @pytest.mark.parametrize('m', Matrices.all_)
  def test_validation(self, m):
    symm = (m == 'K')
    M11 = Files.read_matrix(m, hi=1, he=1, symm=symm)
    M12 = Files.read_matrix(m, hi=1, he=2, symm=symm)
    M21 = Files.read_matrix(m, hi=2, he=1, symm=symm)
    M22 = Files.read_matrix(m, hi=2, he=2, symm=symm)  # 검증용

    if symm:
      assert M11.shape[0] == M11.shape[1]
    else:
      assert M11.shape[1] == 1

    H = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)
    mh = sm.MatrixH(H=H, Ms=[M11, M12, M21])

    M22e = mh.matrix(h_interior=2.0, h_exterior=2.0)

    assert np.allclose(M22.toarray(), M22e.toarray())

  @pytest.mark.parametrize('m', Matrices.all_)
  def test_from_files(self, m):
    symm = (m == 'K')
    M22 = Files.read_matrix(m, hi=2, he=2, symm=symm)

    H = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)

    mh = sm.MatrixH.from_files(H=H,
                               files=[Files.get_path(m, x[0], x[1]) for x in H],
                               is_square=symm,
                               is_symmetric=symm)

    M22e = mh.matrix(h_interior=2.0, h_exterior=2.0)

    assert np.allclose(M22.toarray(), M22e.toarray())


if __name__ == '__main__':
  pytest.main(['-vv', '-k', 'test_system_matrix'])
