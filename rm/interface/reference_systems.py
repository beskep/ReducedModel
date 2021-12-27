from pathlib import Path
from typing import List, Optional

from loguru import logger
import yaml

from rm.reduced_model.thermal_model import StateSpace
from rm.reduced_model.thermal_model import ThermalModel
from rm.utils import DIR


class ReferenceSystems:
  FNAME = 'LinearThermalTransmittance.yaml'

  def __init__(self, path: Optional[Path] = None) -> None:
    if path is None:
      path = DIR.RESOURCE.joinpath('models')

    self._paths = tuple(path.glob('*.npz'))
    self._names = tuple(x.stem for x in self._paths)

    with path.joinpath(self.FNAME).open('r', encoding='utf-8') as f:
      self._psi: dict = yaml.safe_load(f)

    npz = sorted(self._names)
    psi = sorted(self._psi.keys())
    logger.debug('Reference models (npz): {}', npz)
    logger.debug('Reference models (yaml): {}', psi)

    if npz != psi:
      raise ValueError('레퍼런스 모델 파일, 선형열관류율 불일치')

  @property
  def names(self):
    return self._names

  @property
  def paths(self):
    return self._paths

  @property
  def linear_thermal_transmittance(self):
    return self._psi

  def load(self) -> List[StateSpace]:
    tm = ThermalModel(system=None)
    return [tm.load(x) for x in self.paths]
