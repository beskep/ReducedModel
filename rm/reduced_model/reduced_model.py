"""Deprecated"""
from enum import Enum
from typing import Callable, Optional, Tuple

from control.modelsimp import balred
from control.statesp import StateSpace
from loguru import logger
import numpy as np
from rich.progress import track
from scipy.linalg import inv
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse.linalg import inv as sparse_inv

from rm import utils
import rm.reduced_model.matrix_reader as mr
from rm.utils import StrPath


def reduce_model(order: int, A: csc_matrix, B: csc_matrix, J: csc_matrix):
  original_system = StateSpace(A.toarray(), B.toarray(), J.toarray(), 0)

  if not order:
    reduced_system = original_system
    logger.info('Model reduction을 시행하지 않습니다.')
  else:
    with utils.console.status('Reducing Model'):
      reduced_system = balred(sys=original_system, orders=order)

  reduced_order = reduced_system.A.shape[0]
  if A.shape[0] != reduced_order:
    logger.info('모델 리덕션 완료 (dim {} to {})', A.shape[0], reduced_order)

  return reduced_system, reduced_order


class ModelSettingError(ValueError):
  pass


class ModelNotReduced(ValueError):
  pass


class Location(Enum):
  Interior = 1
  Exterior = 2


class ModelReducer:

  def __init__(self, order: int) -> None:
    self._order = order

    self._damping_mtx: np.ndarray = None
    self._stiffness_mtx: np.ndarray = None
    self._internal_load_mtx: np.ndarray = None
    self._external_load_mtx: np.ndarray = None
    self._max_node: Optional[int] = None

    self._internal_temp_fn: Optional[Callable] = None
    self._external_temp_fn: Optional[Callable] = None
    self._internal_fluid_temp: Optional[float] = None
    self._external_fluid_temp: Optional[float] = None

    self._targets: np.ndarray = None

    self._reduced_system: StateSpace = None
    self._reduced_order: Optional[int] = None

  @property
  def order(self):
    return self._order

  @order.setter
  def order(self, value: int):
    self._order = value

  def read_matrices(self,
                    damping: StrPath,
                    stiffness: StrPath,
                    internal_load: StrPath,
                    external_load: StrPath,
                    max_node=0):
    reader = mr.SystemMatricesReader(damping=damping,
                                     stiffness=stiffness,
                                     internal_load=internal_load,
                                     external_load=external_load,
                                     max_node=max_node)
    self._damping_mtx = reader.damping_matrix
    self._stiffness_mtx = reader.stiffness_matrix
    self._internal_load_mtx = reader.internal_load_matrix
    self._external_load_mtx = reader.external_load_matrix
    self._max_node = reader.max_node
    del reader

    for name, mtx in zip(['damping', 'stiffness'],
                         [self._damping_mtx, self._stiffness_mtx]):
      if mtx.shape[0] != mtx.shape[1]:
        raise ValueError(f'{name} matrix가 정방행렬이 아님')

  def has_matrices(self) -> bool:
    matrices = (self._damping_mtx, self._stiffness_mtx, self._internal_load_mtx,
                self._external_load_mtx)
    return all(x is not None for x in matrices)

  def has_reduced_model(self) -> bool:
    return all(
        x is not None for x in [self._reduced_system, self._reduced_order])

  def set_fluid_temperature(self, interior: float, exterior: float):
    self._internal_fluid_temp = interior
    self._external_fluid_temp = exterior

  def set_target_nodes(self, files):
    reader = mr.MatrixReader(path=None, shape=(self._max_node, 1))
    targets = []
    for file in files:
      reader.path = file
      mtx = reader.read_matrix()
      mtx /= mtx.data.size  # 0이 아닌 node 개수로 나눔... 왜인진 모름
      targets.append(mtx)

    self._targets = sparse_hstack(targets)

  def set_temperature_condition(self, fn: Callable[[int], float],
                                loc: Location):
    """
    온도 조건 함수 설정

    Parameters
    ----------
    fn : Callable[[int], float]
        time[step] -> temperature[°C]
    loc : Location
        location
    """
    if not isinstance(loc, Location):
      raise ValueError(f'f{loc} not in {Location}')

    if loc is Location.Interior:
      self._internal_temp_fn = fn
    elif loc is Location.Exterior:
      self._external_temp_fn = fn

  def _check_matrices(self):
    for name, var in zip(['system matrix', 'target nodes'],
                         [self._damping_mtx, self._targets]):
      if var is None:
        raise ModelSettingError(f'{name}이/가 설정되지 않았습니다.')

  def _check_environment_variables(self):
    for name, var in zip(
        [
            'internal air temperature',
            'external air temperature',
            'interior temperature function',
            'exterior temperature function',
        ],
        [
            self._internal_fluid_temp,
            self._external_fluid_temp,
            self._internal_temp_fn,
            self._external_temp_fn,
        ],
    ):
      if var is None:
        raise ModelSettingError(f'{name}이/가 설정되지 않았습니다.')

  def compute_state_matrices(self) -> Tuple[csc_matrix, csc_matrix, csr_matrix]:
    self._check_matrices()

    load_all = sparse_hstack([
        self._internal_load_mtx / self._internal_fluid_temp,
        self._external_load_mtx / self._external_fluid_temp,
    ])

    inv_damping = sparse_inv(self._damping_mtx)  # inv(C)
    A = -np.dot(inv_damping, self._stiffness_mtx)  # inv(C) * -K
    B = np.dot(inv_damping, load_all)  # inv(C) * L
    J = self._targets.transpose()

    return A, B, J

  def reduce_model(self) -> tuple[StateSpace, int]:
    A, B, J = self.compute_state_matrices()
    logger.info('State matrices 계산 완료')

    system, order = reduce_model(self._order, A, B, J)

    self._reduced_system = system
    self._reduced_order = order

    return system, order

  def set_reduced_model(self, reduced_system: StateSpace, reduced_order: int):
    self._reduced_system = reduced_system
    self._reduced_order = reduced_order

  def compute(self,
              dt: float,
              time_step: int,
              initial_temperature: float = None,
              callback: Callable[[np.ndarray], None] = None,
              reduced_system=True) -> np.ndarray:
    """
    Parameters
    ----------
    dt : float
        Delta time [sec]
    time_step : int
        Total time steps
    initial_temperature : float, optional
        Initial temperature [ºC]
    callback : Callable[[np.ndarray], None], optional
        Callback function for each time step (f: results(<np.ndarray>) -> None)
    reduced_system : bool, optional
        If false, use original system

    Returns
    -------
    np.ndarray
        Temperature of each location
        (shape: (time step, number of target locations)).

    Raises
    ------
    ModelNotReduced
        if `reduced_system` is True and the model is not reduced

    References
    ----------
    [1] Choi, J.-S., Kim, C.-M., Jang, H.-I., & Kim, E.-J. (2021).
    Detailed and fast calculation of wall surface temperatures near thermal
    bridge area. Case Studies in Thermal Engineering, 25, 100936.
    https://doi.org/10.1016/j.csite.2021.100936
    """
    self._check_environment_variables()

    if reduced_system:
      if self._reduced_system is None:
        raise ModelNotReduced('모델이 축소되지 않았습니다.')
      ss = self._reduced_system
      order = self._reduced_order
    else:
      A, B, J = self.compute_state_matrices()
      ss = StateSpace(A.toarray(), B.toarray(), J.toarray(), 0)  # pylint: disable=no-member
      order = A.shape[0]

    Omega = inv(np.eye(order) - dt * ss.A)  # inv(eye(Order) - dt * Ar)
    Pi = dt * np.dot(Omega, ss.B)  # dt * Red1 * Br

    # X0
    if initial_temperature is not None:
      Xn = self.initial_x(Omega=Omega,
                          Pi=Pi,
                          T0=initial_temperature,
                          max_iteration=1000,
                          rtol=1e-5,
                          atol=0)
    else:
      Xn = np.zeros(shape=(order, 1))

    # 본 연산
    results = None
    for ts in track(range(time_step), description='Computing...'):
      temperature = np.array([[self._internal_temp_fn(ts)],
                              [self._external_temp_fn(ts)]])

      Xnp1 = np.dot(Omega, Xn) + np.dot(Pi, temperature)
      Y = (np.dot(ss.C, Xnp1) + np.dot(ss.D, temperature))

      Xn = Xnp1
      mtxYrow = Y.reshape([1, -1])
      if results is None:
        results = mtxYrow
      else:
        results = np.vstack((results, mtxYrow))

      if callback is not None:
        callback(results)

    return results

  @staticmethod
  def initial_x(Omega: np.ndarray,
                Pi: np.ndarray,
                T0: float,
                max_iteration=1000,
                rtol=1e-5,
                atol=1e-8) -> np.ndarray:
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
    max_iteration : int, optional
        Max iteration, by default 10000
    rtol : float, optional
        Relative convergence tolerance, by default 1e-5
    atol : float, optional
        Absolute convergence tolerance, by default 1e-8

    Returns
    -------
    np.ndarray
        X0
    """
    T = np.array([[T0], [T0]])
    PiT = np.dot(Pi, T)

    Xn = np.zeros(shape=(Omega.shape[0], 1))
    Xnp1 = Xn.copy()
    for idx in range(max_iteration):
      Xnp1 = np.dot(Omega, Xn) + PiT
      if np.allclose(Xnp1, Xn, rtol=rtol, atol=atol):
        logger.debug('X0 converged after {} iterations (rtol={}, atol={})', idx,
                     rtol, atol)
        break

      Xn = Xnp1

    return Xnp1

  def save_reduced_model(self, path):
    if self._reduced_system is None:
      raise ValueError('Reduced system is None')

    np.savez(path,
             A=self._reduced_system.A,
             B=self._reduced_system.B,
             C=self._reduced_system.C,
             D=self._reduced_system.D,
             order=self._reduced_order)

  def load_reduced_model(self, path):
    npz = np.load(path)

    self._reduced_system = StateSpace(npz['A'], npz['B'], npz['C'], npz['D'])
    self._reduced_order = npz['order']
