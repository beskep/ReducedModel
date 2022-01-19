from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Union

from loguru import logger
import numpy as np
from rich.progress import track
from scipy.linalg import inv

from rm import utils

from .state_space import StateSpace
from .state_space import System
from .state_space import SystemH


@dataclass
class X0Option:
  """X0 계산 수렴 옵션"""
  max_iter: int = 1000
  rtol: float = 1e-5
  atol: float = 0.0


@dataclass
class Matrices:
  omega: np.ndarray
  pi: np.ndarray
  x: np.ndarray


class ThermalModel:
  CALLBACK_UPDATE = 20

  def __init__(self, system: Union[System, SystemH, None]) -> None:
    self._system = system
    self._x0opt = X0Option()

  @property
  def system(self) -> Union[System, SystemH, None]:
    return self._system

  def set_x0_option(self, option: X0Option):
    self._x0opt = option

  def state_space(self,
                  order: Optional[int] = None,
                  hi: Optional[float] = None,
                  he: Optional[float] = None,
                  reduction_method: Optional[str] = 'truncate') -> StateSpace:
    if self._system is None:
      raise ValueError('system is None')

    if isinstance(self._system, SystemH):
      if hi is None or he is None:
        raise ValueError(f'hi: {hi}, he: {he}')

      system = self._system.system(hi=hi, he=he)
    else:
      system = self._system

    ss = system.model(order=order, reduction_method=reduction_method)

    return ss

  def save(self, path, state_space: Optional[StateSpace] = None):
    if state_space is None:
      state_space = self.state_space()

    np.savez(path,
             A=state_space.A,
             B=state_space.B,
             C=state_space.C,
             D=state_space.D)

  @staticmethod
  def load(path):
    npz = np.load(path)
    ss = StateSpace(npz['A'], npz['B'], npz['C'], npz['D'])

    return ss

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

  def _matrices(self, ss: StateSpace, dt, T0):
    order = ss.A.shape[0]
    omega = inv(np.eye(order) - dt * ss.A)
    pi = dt * np.dot(omega, ss.B)

    if T0 is not None:
      x0 = self.inital_x(Omega=omega, Pi=pi, T0=T0)
    else:
      x0 = np.zeros(shape=(order, 1))

    return Matrices(omega=omega, pi=pi, x=x0)

  def compute(self,
              ss: StateSpace,
              dt: float,
              bc: np.ndarray,
              T0: Optional[float] = None,
              callback: Optional[Callable[[np.ndarray], None]] = None,
              progress=True) -> np.ndarray:
    """
    시간별 지정된 모델 노드의 온도 변화 계산

    Parameters
    ----------
    ss : StateSpace
        State space system
    dt : float
        Delta time
    bc : np.ndarray
        Boundary condition (boundary temperature).
        bc[:, 0]은 internal temperature, bc[:, 1]은 external temperature인
        2차원 ndarray
    T0 : Optional[float], optional
        Initial temperature, by default None
    callback : Optional[Callable[[np.ndarray], None]], optional
        매 회 실행하는 callback 함수. 입력 인자는 계산한 온도 행렬.
    progress : bool, optional
        rich.progress 표시 여부

    Returns
    -------
    np.ndarray
        온도 행렬. 각 행이 time step, 열이 target nodes를 의미.
    """
    ms = self._matrices(ss=ss, dt=dt, T0=T0)
    y: Any = None

    it: Iterable = range(bc.shape[0])
    if progress:
      it = track(it, description='Computing...', console=utils.console)

    update = bc.shape[0] // self.CALLBACK_UPDATE

    for step in it:
      bcn = bc[step].reshape([2, 1])  # [[Tint], [Text]]

      yn = np.dot(ss.C, ms.x) + np.dot(ss.D, bcn)
      ms.x = np.dot(ms.omega, ms.x) + np.dot(ms.pi, bcn)

      if y is None:
        y = yn.reshape([1, -1])
      else:
        y = np.vstack((y, yn.reshape([1, -1])))

      if callback and (step % update == 0):
        callback(y)

    if callback:
      callback(y)

    return y

  def compute_multi_systems(
      self,
      sss: List[StateSpace],
      dt: float,
      bc: np.ndarray,
      T0: Optional[float] = None,
      callback: Optional[Callable[[np.ndarray], None]] = None,
      progress=True,
  ) -> np.ndarray:
    """
    다수 모델의 시간별 노드의 온도 변화 계산

    Parameters
    ----------
    sss : List[StateSpace]
        State space systems
    dt : float
        Delta time
    bc : np.ndarray
        Boundary condition (boundary temperature).
        bc[:, 0]은 internal temperature, bc[:, 1]은 external temperature인
        2차원 ndarray
    T0 : Optional[float], optional
        Initial temperature, by default None
    callback : Optional[Callable[[np.ndarray], None]], optional
        매 회 실행하는 callback 함수. 입력 인자는 계산한 온도 행렬.
    progress : bool, optional
        rich.progress 표시 여부

    Returns
    -------
    np.ndarray
        온도 행렬. 각 행이 time step, 열이 모델/target nodes를 의미.
    """
    matrices = [self._matrices(ss=ss, dt=dt, T0=T0) for ss in sss]
    y: Any = None

    it: Iterable = range(bc.shape[0])
    if progress:
      it = track(it, description='Computing...', console=utils.console)

    update = bc.shape[0] // self.CALLBACK_UPDATE

    for step in it:
      bcn = bc[step].reshape([2, 1])  # [[Tint], [Text]]

      yns = []
      for ss, ms in zip(sss, matrices):
        ynm = np.dot(ss.C, ms.x) + np.dot(ss.D, bcn)
        ms.x = np.dot(ms.omega, ms.x) + np.dot(ms.pi, bcn)

        yns.append(ynm.ravel())

      # [Y(model0, point0), Y(model0, point1), ...,
      #  Y(model1, point0), Y(model1, point1), ...]
      # (ndim == 1)
      yn = np.hstack(yns)

      if y is None:
        y = yn
      else:
        y = np.vstack((y, yn))

      if callback and (step % update == 0):
        callback(y)

    if callback:
      callback(y)

    return y
