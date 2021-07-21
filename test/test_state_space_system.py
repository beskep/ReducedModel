import context
import utils

import numpy as np
import pytest
from files import Files, Matrices, data_dir

import reduced_model.state_space_system as sss

pathC = data_dir.joinpath('C_o.txt')
pathNs = [data_dir.joinpath('specific1_node.txt')]


def _system(Ti, Te):
  sys = sss.System.from_files(C=pathC,
                              K=Files.get_path(m=Matrices.K, hi=2, he=2),
                              Li=Files.get_path(m=Matrices.Li, hi=2, he=2),
                              Le=Files.get_path(m=Matrices.Le, hi=2, he=2),
                              Ti=Ti,
                              Te=Te,
                              Ns=pathNs)
  return sys


def _system_h(Ti, Te):
  H = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)

  K = [Files.get_path(Matrices.K, x[0], x[1]) for x in H]
  Li = [Files.get_path(Matrices.Li, x[0], x[1]) for x in H]
  Le = [Files.get_path(Matrices.Le, x[0], x[1]) for x in H]

  system_h = sss.SystemH.from_files(H=H,
                                    C=pathC,
                                    K=K,
                                    Li=Li,
                                    Le=Le,
                                    Ti=Ti,
                                    Te=Te,
                                    Ns=pathNs)

  sysh = system_h.system(hi=2, he=2)

  return sysh


@pytest.mark.parametrize(['Ti', 'Te'], [(0.2, 1.25)])
def test_h(Ti, Te):
  sys = _system(Ti, Te)
  sysh = _system_h(Ti, Te)

  assert np.allclose(sys.C.toarray(), sysh.C.toarray())
  assert np.allclose(sys.K.toarray(), sysh.K.toarray())
  assert np.allclose(sys.LiTi.toarray(), sysh.LiTi.toarray())
  assert np.allclose(sys.LeTe.toarray(), sysh.LeTe.toarray())

  for N, Nh in zip(sys.Ns, sysh.Ns):
    assert np.allclose(N.toarray(), Nh.toarray())

  # reduce
  ss = sys.model(order=5)
  ssh = sysh.model(order=5)

  assert ss.A == pytest.approx(ssh.A)
  assert ss.B == pytest.approx(ssh.B)
  assert ss.C == pytest.approx(ssh.C)


if __name__ == '__main__':
  pytest.main(['-vv', '-k', 'test_state_space_system'])
