from typing import List

import numpy as np
from scipy.sparse import csc_matrix


class MatrixH:

  def __init__(self, H: np.ndarray, Ms: List[csc_matrix]) -> None:
    """
    M(hi, he)

    다음 식에 따라 M0, Mint, Mext를 추정하고,
    내외부 하자 부위 대류열전달계수에 따라 M 산정.

    계산에 세 케이스의 행렬이 필요함.

    M(h_int, h_ext) = M0 + h_int * Mint + h_ext * Mext

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

        `hi`: interior h (대류열전달계수)
        `he`: exterior h

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
    self.Mint: csc_matrix = sum((invH[0, x] * Ms[x] for x in range(3)))
    self.Mext: csc_matrix = sum((invH[1, x] * Ms[x] for x in range(3)))
    self.M0: csc_matrix = sum((invH[2, x] * Ms[x] for x in range(3)))

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

  def matrix(self, h_interior: float, h_exterior: float) -> csc_matrix:
    return self.M0 + (h_interior * self.Mint) + (h_exterior * self.Mext)
