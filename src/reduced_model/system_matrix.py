from typing import List

from utils import StrPath

import numpy as np
from scipy.sparse import csc_matrix

from .matrix_reader import MatricesReader


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
