from pathlib import Path

import context
import utils

import numpy as np

from reduced_model.reduced_model import Location, ModelReducer


def compute(orders):
  root_dir = Path(__file__).parents[1]
  save_dir = root_dir.joinpath('res')

  for order in orders:
    model = ModelReducer(order=order)

    model.read_matrices(damping=root_dir.joinpath('res/C.txt'),
                        stiffness=root_dir.joinpath('res/K.txt'),
                        internal_load=root_dir.joinpath('res/Lin.txt'),
                        external_load=root_dir.joinpath('res/Lout.txt'))

    target_files = [
        root_dir.joinpath('res/specific1.txt'),
        root_dir.joinpath('res/specific2.txt'),
        root_dir.joinpath('res/specific3.txt'),
    ]
    model.set_target_nodes(target_files)

    model.set_fluid_temperature(interior=20.0, exterior=10.0)

    model.set_temperature_condition(fn=lambda x:
                                    (20 + np.sin(np.pi * 0.5 * x / 6)),
                                    loc=Location.Interior)
    model.set_temperature_condition(fn=lambda x:
                                    (10 + 5 * np.sin(np.pi * 0.5 * x / 6)),
                                    loc=Location.Exterior)

    fname = save_dir.joinpath('reduction_output_{}.txt'.format(order))
    model.compute(dt=3600, time_step=200, fname=fname)


if __name__ == '__main__':
  orders = [None, 50, 20, 10, 5, 2, 1]
  compute(orders=orders)
