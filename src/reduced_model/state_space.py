from dataclasses import dataclass
from itertools import chain
from typing import List, Optional, Tuple, Union

import utils
from utils import StrPath

import numpy as np
from control.modelsimp import balred
from control.statesp import StateSpace
from loguru import logger
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as sparse_inv

from .matrix_reader import MatricesReader, read_matrix


def _nodes(matrix: Union[csc_matrix, StrPath],
           reader: Optional[MatricesReader] = None) -> csc_matrix:
  if not isinstance(matrix, csc_matrix):
    if reader is None:
      matrix = read_matrix(path=matrix, symmetric=False)
    else:
      matrix = reader.read_matrix(path=matrix, square=False)

  matrix /= matrix.data.size  # 0이 아닌 node 개수로 나눔. 왜인지는 모름...

  return matrix


def _state_space(A: csc_matrix, B: csc_matrix, J: csc_matrix):
  return StateSpace(A.toarray(), B.toarray(), J.toarray(), 0.0)


def reduce_model(A: csc_matrix, B: csc_matrix, J: csc_matrix,
                 order: int) -> StateSpace:
  ss = _state_space(A, B, J)

  with utils.console.status('Reducing model'):
    reduced_system: StateSpace = balred(sys=ss, orders=order)

  logger.info(
      '모델 리덕션 완료 (dim {} to {})',
      A.shape[0],
      reduced_system.A.shape[0],
  )

  return reduced_system


class MatrixH:

  def __init__(self, H: np.ndarray, Ms: List[csc_matrix]) -> None:
    """
    M(hi, he)

    다음 식에 따라 M0, Mint, Mext를 추정하고,
    내외부 하자 부위 대류열전달계수에 따라 M 산정.

    계산에 세 케이스의 행렬이 필요함.

    M(hi, he) = M0 + (hi * Mi) + (he * Me)
    hi: interior h (대류열전달계수)
    he: exterior h
    Mi: 행렬의 hi 성분
    Me: 행렬의 he 성분
    M0: hi, he와 독립적인 나머지 성분

    Parameters
    ----------
    H : np.ndarray

            `[[hi_1, he_1, 1.0],`

            ` [hi_2, he_2, 1.0],`

            ` [hi_3, he_3, 1.0]]`

        또는

            `[[hi_1, he_1],`

            ` [hi_2, he_2],`

            ` [hi_3, he_3]]`


    Ms : List[csc_matrix]
        h에 상응하는 M 행렬 목록
        [M1, M2, M3]
    """
    H = self._checkH(H)

    if len(Ms) != 3:
      raise ValueError

    try:
      invH = np.linalg.inv(H)
    except np.linalg.LinAlgError as e:
      raise np.linalg.LinAlgError(f'{e}: 주어진 H로 행렬을 추정할 수 없습니다.')

    # (invH[i, 1] * M1) + (invH[i, 2] * M2) + (invH[i, 3] * M3)
    self._Mi: csc_matrix = sum((invH[0, x] * Ms[x] for x in range(3)))
    self._Me: csc_matrix = sum((invH[1, x] * Ms[x] for x in range(3)))
    self._M0: csc_matrix = sum((invH[2, x] * Ms[x] for x in range(3)))

  @staticmethod
  def _checkH(H: np.ndarray):
    if H.ndim != 2:
      raise ValueError
    if H.shape[0] != 3:
      raise ValueError

    if H.shape[1] == 2:
      H = np.hstack((
          H,
          np.ones((3, 1)),
      ))
    elif H.shape[1] != 3:
      raise ValueError

    return H

  @classmethod
  def from_files(
      cls,
      H: np.ndarray,
      files: List[StrPath],
      square: bool,
      symmetric: bool,
  ):
    if len(files) != 3:
      raise ValueError

    mr = MatricesReader(files=files)
    Ms = [mr.read_matrix(f, square=square, symmetric=symmetric) for f in files]

    return cls(H=H, Ms=Ms)

  def matrix(self, hi: float, he: float) -> csc_matrix:
    """
    실내외 대류열전달계수에 따른 행렬 계산

    Parameters
    ----------
    hi : float
        Interior h
    he : float
        Exterior h

    Returns
    -------
    csc_matrix
    """
    return self._M0 + (hi * self._Mi) + (he * self._Me)

  def set_fluid_temperature(self, temperature: float):
    self._M0 /= temperature
    self._Mi /= temperature
    self._Me /= temperature


@dataclass
class System:
  """
  State Space System

  Parameters
  ----------
  C: csc_matrix
      Capacitance/Damping matrix
  K: csc_matrix
      Conductance/Stiffness matrix
  LiTi: csc_matrix
      Internal Load (Solicitation) matrix / fluid temperature
  LeTe: csc_matrix
      External Load (Solicitation) matrix / fluid temperature
  Ns: List[csc_matrix]
      Target nodes matrix
  """
  C: csc_matrix  # Capacitance/Damping
  K: csc_matrix  # Conductance/Stiffness
  LiTi: csc_matrix  # Internal Load (Solicitation) / fluid temperature
  LeTe: csc_matrix  # External Load (Solicitation) / fluid temperature
  Ns: List[csc_matrix]  # Target nodes

  @classmethod
  def from_files(
      cls,
      C: StrPath,
      K: StrPath,
      Li: StrPath,
      Le: StrPath,
      Ti: float,
      Te: float,
      Ns: List[StrPath],
  ):
    mr = MatricesReader(files=[C, K, Li, Le] + Ns)

    C_ = mr.read_matrix(path=C, square=True)
    K_ = mr.read_matrix(path=K, square=True, symmetric=True)
    Li_ = mr.read_matrix(path=Li, square=False)
    Le_ = mr.read_matrix(path=Le, square=False)
    Ns_ = [_nodes(matrix=x, reader=mr) for x in Ns]

    return cls(C=C_, K=K_, LiTi=(Li_ / Ti), LeTe=(Le_ / Te), Ns=Ns_)

  def state_matrices(self) -> Tuple[csc_matrix, csc_matrix, csc_matrix]:
    L = sparse.hstack([self.LiTi, self.LeTe])

    invC = sparse_inv(self.C)  # inv(C)
    A = -np.dot(invC, self.K)  # inv(C) * -K
    B = np.dot(invC, L)  # inv(C) * L
    J = sparse.hstack(self.Ns).transpose()

    return A, B, J

  def model(self, order: Optional[int] = None):
    A, B, J = self.state_matrices()

    if order is None:
      ss = _state_space(A, B, J)
    elif order < A.shape[0]:
      ss = reduce_model(A=A, B=B, J=J, order=order)
    else:
      logger.warning('지정한 차수 ({})가 모델 차수 ({}) 이상입니다. '
                     '모델을 축소하지 않습니다.', order, A.shape[0])
      ss = _state_space(A, B, J)

    return ss


@dataclass
class SystemH:
  """
  State Space System by hi, he (internal, external h)

  Parameters
  ----------
  C: csc_matrix
      Capacitance/Damping matrix
  K: MatrixH
      Conductance/Stiffness matrix
  LiTi: MatrixH
      Internal Load (Solicitation) matrix / fluid temperature
  LeTe: MatrixH
      External Load (Solicitation) matrix / fluid temperature
  Ns: List[csc_matrix]
      Target nodes matrix
  """
  C: csc_matrix  # Capacitance/Damping
  K: MatrixH  # Conductance/Stiffness
  LiTi: MatrixH  # Internal Load (Solicitation) / fluid temperature
  LeTe: MatrixH  # External Load (Solicitation) / fluid temperature
  Ns: List[csc_matrix]  # Target nodes

  @classmethod
  def from_files(
      cls,
      H: np.ndarray,
      C: StrPath,
      K: List[StrPath],
      Li: List[StrPath],
      Le: List[StrPath],
      Ti: float,
      Te: float,
      Ns: List[StrPath],
  ):
    mr = MatricesReader(files=chain([C], K, Li, Le))

    Ks = [mr.read_matrix(path=x, square=True, symmetric=True) for x in K]
    K_ = MatrixH(H=H, Ms=Ks)

    LiTi = MatrixH(H=H, Ms=[mr.read_matrix(path=x, square=False) for x in Li])
    LiTi.set_fluid_temperature(Ti)

    LeTe = MatrixH(H=H, Ms=[mr.read_matrix(path=x, square=False) for x in Le])
    LeTe.set_fluid_temperature(Te)

    C_ = mr.read_matrix(C, square=True)
    Ns_ = [_nodes(matrix=x, reader=mr) for x in Ns]

    return cls(C=C_, K=K_, LiTi=LiTi, LeTe=LeTe, Ns=Ns_)

  def system(self, hi: float, he: float) -> System:
    K = self.K.matrix(hi=hi, he=he)
    LiTi = self.LiTi.matrix(hi=hi, he=he)
    LeTe = self.LeTe.matrix(hi=hi, he=he)

    system = System(C=self.C, K=K, LiTi=LiTi, LeTe=LeTe, Ns=self.Ns)

    return system
