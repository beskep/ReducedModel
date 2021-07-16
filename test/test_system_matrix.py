import context
import utils

import numpy as np
import pytest

import reduced_model.system_matrix as sm

from test_h import read_K


def test_inv_error():
  with pytest.raises(np.linalg.LinAlgError):
    sm.MatrixH(H=np.ones([3, 3]), Ms=[None, None, None])


def test_validation():
  K11 = read_K(1, 1)
  K12 = read_K(1, 2)
  K21 = read_K(2, 1)
  K22 = read_K(2, 2)

  H = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)
  mh = sm.MatrixH(H=H, Ms=[K11, K12, K21])

  K22e = mh.matrix(h_interior=2.0, h_exterior=2.0)

  assert np.allclose(K22.toarray(), K22e.toarray())


if __name__ == '__main__':
  pytest.main(['-vv', '-k', 'test_system_matrix'])
