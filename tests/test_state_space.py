import numpy as np
import pytest
from test_case import data_dir
from test_case import Files
from test_case import Matrices

from rm import utils
from rm.reduced_model.state_space import MatrixH
from rm.reduced_model.state_space import System
from rm.reduced_model.state_space import SystemH

pathC = data_dir.joinpath('C_o.txt')
pathNs = [data_dir.joinpath('specific1_node.txt')]


class TestMatrixHHypothesis:

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
      MatrixH(H=np.ones([3, 3]), Ms=[None, None, None])

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
    mh = MatrixH(H=H, Ms=[M11, M12, M21])

    M22e = mh.matrix(hi=2.0, he=2.0)

    assert np.allclose(M22.toarray(), M22e.toarray())

  @pytest.mark.parametrize('m', Matrices.all_)
  def test_from_files(self, m):
    symm = (m == 'K')
    M22 = Files.read_matrix(m, hi=2, he=2, symm=symm)

    H = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)

    mh = MatrixH.from_files(H=H,
                            files=[Files.get_path(m, x[0], x[1]) for x in H],
                            square=symm,
                            symmetric=symm)

    M22e = mh.matrix(hi=2.0, he=2.0)

    assert np.allclose(M22.toarray(), M22e.toarray())


def _system(Ti, Te):
  sys = System.from_files(C=pathC,
                          K=Files.get_path(m=Matrices.K, hi=2, he=2),
                          Li=Files.get_path(m=Matrices.Li, hi=2, he=2),
                          Le=Files.get_path(m=Matrices.Le, hi=2, he=2),
                          Ti=Ti,
                          Te=Te,
                          Ns=pathNs)
  return sys


def _system_h(Ti, Te):
  H = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)

  K = [Files.get_path(Matrices.K, x[0], x[1]) for x in H]
  Li = [Files.get_path(Matrices.Li, x[0], x[1]) for x in H]
  Le = [Files.get_path(Matrices.Le, x[0], x[1]) for x in H]

  system_h = SystemH.from_files(H=H,
                                C=pathC,
                                K=K,
                                Li=Li,
                                Le=Le,
                                Ti=Ti,
                                Te=Te,
                                Ns=pathNs)

  sysh = system_h.system(hi=2, he=2)

  return sysh


@pytest.mark.skip()
@pytest.mark.parametrize(['Ti', 'Te'], [(0.2, 1.25)])
def test_h(Ti, Te):
  sys = _system(Ti, Te)
  sysh = _system_h(Ti, Te)

  assert np.allclose(sys.C.toarray(), sysh.C.toarray())
  assert np.allclose(sys.K.toarray(), sysh.K.toarray())
  assert np.allclose(sys.LiTi.toarray(), sysh.LiTi.toarray())
  assert np.allclose(sys.LeTe.toarray(), sysh.LeTe.toarray())

  for N, Nh in zip(sys.Ns, sysh.Ns):
    assert np.allclose(N.toarray(), Nh.toarray())

  # reduce
  ss = sys.model(order=5)
  ssh = sysh.model(order=5)

  assert ss.A == pytest.approx(ssh.A)
  assert ss.B == pytest.approx(ssh.B)
  assert ss.C == pytest.approx(ssh.C)
