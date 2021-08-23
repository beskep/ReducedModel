"""FAIL"""

import numpy as np
from test_case import data_dir
from test_case import make_system_h

from rm import utils
from rm.reduced_model.optimize import ConstOptions
from rm.reduced_model.optimize import ThermalModelOptimizer


def optimize():
  systemh = make_system_h()

  Tint = np.loadtxt(data_dir.joinpath('data_U1.txt'))
  Text = np.loadtxt(data_dir.joinpath('data_U2.txt'))
  bc = np.vstack((Tint, Text)).T

  const_opts = ConstOptions(
      dt=30 * 60.0,
      T0=20.0,
      order=20,
  )

  y = np.array([
      [1508, 21.65],  # [step, temperature]
      # [2947, 20.95],
      [4375, 20.325],
  ])

  optimizer = ThermalModelOptimizer(system=systemh)

  res = optimizer.optimize(bc=bc,
                           y=y,
                           h0=[1.6649e+00, 1.4802e+01],
                           bounds=(
                               (1.0, 1.0),
                               np.inf,
                           ),
                           opts=const_opts)

  utils.console.print(res)  # FAIL


if __name__ == '__main__':
  utils.set_logger()

  optimize()
