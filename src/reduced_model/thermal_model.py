from dataclasses import dataclass
from typing import Callable, Optional, Union

import utils

import numpy as np
from loguru import logger
from rich.progress import track
from scipy.linalg import inv

from .state_space_system import StateSpace, System, SystemH


@dataclass
class X0Option:
  """X0 계산 수렴 옵션"""
  max_iter: int = 1000
  rtol: float = 1e-5
  atol: float = 0.0


class ThermalModel:
  # TODO save, load 함수

  def __init__(self, system: Union[System, SystemH]) -> None:
    if isinstance(system, System):
      self._system: Optional[System] = system
      self._system_h: Optional[SystemH] = None
    elif isinstance(system, SystemH):
      self._system = None
      self._system_h = system
    else:
      raise ValueError

    self._x0opt = X0Option()

  def set_x0_option(self, option: X0Option):
    self._x0opt = option

  def model(self,
            order: Optional[int] = None,
            hi: Optional[float] = None,
            he: Optional[float] = None) -> StateSpace:
    if self._system_h is not None:
      if hi is None or he is None:
        raise ValueError

      system = self._system_h.system(hi=hi, he=he)
    else:
      assert self._system is not None
      system = self._system

    model = system.model(order=order)

    return model

  def inital_x(self, Omega: np.ndarray, Pi: np.ndarray, T0: float):
    """
    초기 온도 (T0)에 대응하는 reduced model의 X0 계산

    Parameters
    ----------
    Omega : np.ndarray
        Reduced model 선형 방정식의 Ω
    Pi : np.ndarray
        Reduced model 선형 방정식의 Π
    T0 : float
        초기 온도

    Returns
    -------
    np.ndarray
    """
    T = np.array([[T0], [T0]])
    PiT = np.dot(Pi, T)

    Xn = np.zeros(shape=(Omega.shape[0], 1))
    Xnp1 = Xn.copy()
    for idx in range(self._x0opt.max_iter):
      Xnp1 = np.dot(Omega, Xn) + PiT

      if np.allclose(Xnp1, Xn, rtol=self._x0opt.rtol, atol=self._x0opt.atol):
        logger.debug('X0 converged after {} iterations (rtol = {}, atol = {})',
                     idx, self._x0opt.rtol, self._x0opt.atol)
        break

      Xn = Xnp1

    return Xnp1

  def compute(
      self,
      model: StateSpace,
      dt: float,
      bc: np.ndarray,
      T0: Optional[float] = None,
      callback: Optional[Callable[[np.ndarray], None]] = None) -> np.ndarray:
    """
    시간별 지정된 모델 노드의 온도 변화 계산

    Parameters
    ----------
    model : StateSpace
    dt : float
        Delta time
    bc : np.ndarray
        Boundary condition (boundary temperature)
        bc[:, 0]은 internal temperature, bc[:, 1]은 external temperature인
        2차원 ndarray
    T0 : Optional[float], optional
        Initial temperature, by default None
    callback : Optional[Callable[[np.ndarray], None]], optional
        매 회 실행하는 callback 함수. 입력 인자는 계산한 온도 행렬

    Returns
    -------
    np.ndarray
        온도 행렬. 각 행이 time step, 열이 target nodes를 의미.
    """
    order = model.A.shape[0]

    Omega = inv(np.eye(order) - dt * model.A)
    Pi = dt * np.dot(Omega, model.B)

    # X0
    if T0 is not None:
      Xn = self.inital_x(Omega=Omega, Pi=Pi, T0=T0)
    else:
      Xn = np.zeros(shape=(order, 1))

    # 본 연산
    Ystack = None

    for idx in track(range(bc.shape[0]),
                     description='Computing...',
                     console=utils.console):
      # boundary: [[internal temperature],
      #            [external temperature]]
      boundary = bc[idx].reshape([2, 1])

      Yn = np.dot(model.C, Xn) + np.dot(model.D, boundary)
      Xn = np.dot(Omega, Xn) + np.dot(Pi, boundary)

      if Ystack is None:
        Ystack = Yn.reshape([1, -1])
      else:
        Ystack = np.vstack((Ystack, Yn.reshape([1, -1])))

      if callback is not None:
        callback(Ystack)

    return Ystack
