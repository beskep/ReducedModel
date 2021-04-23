import os
from enum import Enum
from typing import Callable, Tuple, Union

import numpy as np
from control.modelsimp import balred
from control.statesp import StateSpace
from loguru import logger
from rich.console import Console
from rich.progress import track
from scipy.linalg import inv
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse.linalg import inv as sparse_inv

import reduced_model.matrix_reader as mr

cnsl = Console()


def reduce_model(order: int, mtxA: csc_matrix, mtxB: csc_matrix,
                 mtxJ: csc_matrix):
  original_system = StateSpace(mtxA.toarray(), mtxB.toarray(), mtxJ.toarray(),
                               0)

  if not order:
    reduced_system = original_system
    logger.info('Model reduction을 시행하지 않습니다.')
  else:
    with cnsl.status('Reducing Model'):
      reduced_system = balred(sys=original_system, orders=order)

  reduced_order = reduced_system.A.shape[0]
  if mtxA.shape[0] != reduced_order:
    logger.info('모델 리덕션 완료 (dim {} to {})', mtxA.shape[0], reduced_order)

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
    self._max_node: int = None

    self._internal_temp_fn = None
    self._external_temp_fn = None
    self._interior_fluid_temp: float = None
    self._exterior_fluid_temp: float = None

    self._targets: np.ndarray = None

    self._reduced_system: StateSpace = None
    self._reduced_order: int = None

  @property
  def order(self):
    return self._order

  @order.setter
  def order(self, value: int):
    self._order = value

  def read_matrices(
      self,
      damping: Union[str, bytes, os.PathLike],
      stiffness: Union[str, bytes, os.PathLike],
      internal_load: Union[str, bytes, os.PathLike],
      external_load: Union[str, bytes, os.PathLike],
      max_node=None,
  ):
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
    return all(x is not None for x in [
        self._damping_mtx, self._stiffness_mtx, self._internal_load_mtx,
        self._external_load_mtx
    ])

  def has_reduced_model(self) -> bool:
    return all(
        x is not None for x in [self._reduced_system, self._reduced_order])

  def set_fluid_temperature(self, interior: float, exterior: float):
    self._interior_fluid_temp = interior
    self._exterior_fluid_temp = exterior

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
            'fluid temperature',
            'interior temperature function',
            'exterior temperature function',
        ],
        [
            self._interior_fluid_temp,
            self._internal_temp_fn,
            self._external_temp_fn,
        ],
    ):
      if var is None:
        raise ModelSettingError(f'{name}이/가 설정되지 않았습니다.')

  def compute_state_matrices(self) -> Tuple[csc_matrix, csc_matrix, csr_matrix]:
    self._check_matrices()

    load_all = sparse_hstack([
        self._internal_load_mtx / self._interior_fluid_temp,
        self._external_load_mtx / self._exterior_fluid_temp,
    ])

    inv_damping = sparse_inv(self._damping_mtx)  # inv(C)
    mtxA = -np.dot(inv_damping, self._stiffness_mtx)  # inv(C) * -K
    mtxB = np.dot(inv_damping, load_all)  # inv(C) * L
    mtxJ = self._targets.transpose()

    return mtxA, mtxB, mtxJ

  def reduce_model(self) -> tuple[StateSpace, int]:
    mtxA, mtxB, mtxJ = self.compute_state_matrices()
    logger.info('State matrices 계산 완료')

    rsystem, rorder = reduce_model(self._order, mtxA, mtxB, mtxJ)

    self._reduced_system = rsystem
    self._reduced_order = rorder

    return rsystem, rorder

  def set_reduced_model(self, reduced_system: StateSpace, reduced_order: int):
    self._reduced_system = reduced_system
    self._reduced_order = reduced_order

  def compute(self,
              dt: float,
              time_step: int,
              initial_temperature=0.0,
              callback: Callable[[np.ndarray], None] = None,
              reduced_system=True) -> np.ndarray:
    """
    Parameters
    ----------
    dt : float
        delta time [sec]
    time_step : int
        total time steps
    initial_temperature : float, optional
        initial temperature [ºC], by default 0.0
    callback : Callable[[np.ndarray], None], optional
        callback function for each time step (f: results(<np.ndarray>) -> None)
    reduced_system : bool, optional
        if false, use original system

    Returns
    -------
    np.ndarray
        temperature of each location (shape: (time step, number of target locations))

    Raises
    ------
    ModelNotReduced
        if `reduced_system` is True and the model is not reduced

    References
    ----------
    [1] Choi, J.-S., Kim, C.-M., Jang, H.-I., & Kim, E.-J. (2021).
    Detailed and fast calculation of wall surface temperatures near thermal bridge area.
    Case Studies in Thermal Engineering, 25, 100936. https://doi.org/10.1016/j.csite.2021.100936
    """
    if reduced_system:
      if self._reduced_system is None:
        raise ModelNotReduced('모델이 축소되지 않았습니다.')
      ss = self._reduced_system
      order = self._reduced_order
    else:
      mtxA, mtxB, mtxJ = self.compute_state_matrices()
      ss = StateSpace(mtxA.toarray(), mtxB.toarray(), mtxJ.toarray(), 0)  # pylint: disable=no-member
      order = mtxA.shape[0]

    self._check_environment_variables()

    mtxOmega = inv(np.eye(order) - dt * ss.A)  # inv(eye(Order) - dt * Ar)
    mtxPi = dt * np.dot(mtxOmega, ss.B)  # dt * Red1 * Br

    # 본 연산
    results = None
    mtxXn = np.full(shape=(order, 1), fill_value=initial_temperature)
    for ts in track(range(time_step), description='Computing...'):
      temperature = np.array([[self._internal_temp_fn(ts)],
                              [self._external_temp_fn(ts)]])

      mtxXnp1 = np.dot(mtxOmega, mtxXn) + np.dot(mtxPi, temperature)
      mtxY = (np.dot(ss.C, mtxXnp1) + np.dot(ss.D, temperature))

      mtxXn = mtxXnp1
      mtxYrow = mtxY.reshape([1, -1])
      if results is None:
        results = mtxYrow
      else:
        results = np.vstack((results, mtxYrow))

      if callback is not None:
        callback(results)

    return results

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
