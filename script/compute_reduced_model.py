import context
import utils

import numpy as np

from reduced_model import Location, ModelReducer

if __name__ == '__main__':
  model = ModelReducer(order=50)
  root_dir = utils.ROOT_DIR

  model.read_matrices(
      damping=root_dir.joinpath('data/test_case/C.txt'),
      stiffness=root_dir.joinpath('data/test_case/K.txt'),
      internal_load=root_dir.joinpath('data/test_case/Lin.txt'),
      external_load=root_dir.joinpath('data/test_case/Lout.txt'))

  target_files = [
      root_dir.joinpath('data/test_case/specific1.txt'),
      root_dir.joinpath('data/test_case/specific2.txt'),
      root_dir.joinpath('data/test_case/specific3.txt'),
  ]
  model.set_target_nodes(target_files)

  model.set_fluid_temperature(interior=20.0, exterior=10.0)

  model.set_temperature_condition(fn=lambda x:
                                  (20 + np.sin(np.pi * 0.5 * x / 6)),
                                  loc=Location.Interior)
  model.set_temperature_condition(fn=lambda x:
                                  (10 + 5 * np.sin(np.pi * 0.5 * x / 6)),
                                  loc=Location.Exterior)

  model.reduce_model()

  model.compute(dt=3600, time_step=200, initial_temperature=10.0)
