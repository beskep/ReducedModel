from pathlib import Path

import numpy as np

from rm.reduced_model.reduced_model import Location
from rm.reduced_model.reduced_model import ModelReducer


def compute(orders):
  data_dir = Path(__file__).parents[1].joinpath('sample')
  save_dir = data_dir

  for order in orders:
    model = ModelReducer(order=order)

    model.read_matrices(damping=data_dir.joinpath('DMPV1.txt'),
                        stiffness=data_dir.joinpath('STIF1.txt'),
                        internal_load=data_dir.joinpath('LOAD1int.txt'),
                        external_load=data_dir.joinpath('LOAD1ext.txt'))

    target_files = [
        data_dir.joinpath('target1.txt'),
        data_dir.joinpath('target2.txt'),
    ]
    model.set_target_nodes(target_files)

    model.set_fluid_temperature(interior=20.0, exterior=10.0)

    model.set_temperature_condition(fn=lambda x:
                                    (20 + np.sin(np.pi * 0.5 * x / 6)),
                                    loc=Location.Interior)
    model.set_temperature_condition(fn=lambda x:
                                    (10 + 5 * np.sin(np.pi * 0.5 * x / 6)),
                                    loc=Location.Exterior)

    model.order = order
    model.reduce_model()
    res = model.compute(dt=3600, time_step=200)
    print(res)


if __name__ == '__main__':
  orders = [None, 50, 20, 10, 5, 2, 1]
  orders = [None]
  compute(orders=orders)
