from pathlib import Path

import context
import utils

import numpy as np
import pytest

from reduced_model.reduced_model import Location, ModelReducer


def _read_test_matrices():
  root_dir = Path(__file__).parents[1]

  model = ModelReducer(order=10)

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

  return model.compute_state_matrices()


def test_validate_matrix():
  mtxA, mtxB, mtxJ = _read_test_matrices()

  data_dir = utils.ROOT_DIR.joinpath('data/test_case')
  mtxAmatlab = np.loadtxt(data_dir.joinpath('matlab_A.txt').as_posix(),
                          delimiter=',')
  mtxBmatlab = np.loadtxt(data_dir.joinpath('matlab_B.txt').as_posix(),
                          delimiter=',')
  mtxJmatlab = np.loadtxt(data_dir.joinpath('matlab_J.txt').as_posix(),
                          delimiter=',')

  assert np.allclose(mtxA.toarray(), mtxAmatlab, rtol=0, atol=1e-6)
  assert np.allclose(mtxB.toarray(), mtxBmatlab, rtol=0, atol=1e-6)
  assert np.allclose(mtxJ.toarray(), mtxJmatlab, rtol=0, atol=1e-6)


if __name__ == '__main__':
  pytest.main([])
