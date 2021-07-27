"""Thermal model 오류가 안나는지만 검사"""

import context
import utils

import matplotlib.pyplot as plt
import numpy as np
from files import Files, Matrices, data_dir

from reduced_model import state_space_system as sss
from reduced_model import thermal_model as tm

pathC = data_dir.joinpath('C_o.txt')
pathNs = [data_dir.joinpath('specific1_node.txt')]


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


def make_system_h():
  H = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)

  K = [Files.get_path(Matrices.K, x[0], x[1]) for x in H]
  Li = [Files.get_path(Matrices.Li, x[0], x[1]) for x in H]
  Le = [Files.get_path(Matrices.Le, x[0], x[1]) for x in H]

  system_h = sss.SystemH.from_files(H=H,
                                    C=pathC,
                                    K=K,
                                    Li=Li,
                                    Le=Le,
                                    Ti=20.0,
                                    Te=5.0,
                                    Ns=pathNs)

  return system_h


def test_compute_model():
  temperature = make_temperature()

  sysh = make_system_h()
  # sys = sysh.system(hi=1.0, he=0.7)

  thermal_model = tm.ThermalModel(system=sysh)
  model = thermal_model.model(order=20, hi=2.0, he=2.0)

  T = thermal_model.compute(model=model, dt=3600, bc=temperature, T0=10.0)

  return T


if __name__ == '__main__':
  utils.set_logger(level='DEBUG')

  T = test_compute_model()

  plt.plot(T)
  plt.show()
