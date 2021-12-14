"""Thermal model 오류가 안나는지만 검사"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rm import utils
from rm.reduced_model.thermal_model import ThermalModel

from .test_case import data_dir
from .test_case import make_system_h


def make_temperature():
  Tir = (15.0, 25.0)
  Ter = (5.0, 15.0)

  steps = 100
  dt = 3600.0  # [sec]

  k = 2 * np.pi / (24 * 3600 / dt)
  sin = np.sin(k * np.arange(0, steps))

  Ti = np.average(Tir) + (Tir[1] - Tir[0]) * sin
  Te = np.average(Ter) + (Ter[1] - Ter[0]) * sin

  T = np.vstack((Ti, Te)).T

  return T


@pytest.mark.slow
def test_compute_model():
  sysh = make_system_h()

  thermal_model = ThermalModel(system=sysh)
  model = thermal_model.state_space(order=10, hi=1.665, he=14.802)

  temperature = make_temperature()
  T = thermal_model.compute(ss=model, dt=3600, bc=temperature, T0=20.0)

  return T


if __name__ == '__main__':
  utils.set_logger(level='DEBUG')

  T = test_compute_model()

  plt.plot(T)
  plt.show()
