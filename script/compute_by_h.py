import numpy as np
import pandas as pd

from rm import utils
from rm.reduced_model import state_space as sss
from rm.reduced_model import thermal_model as tm

data_dir = utils.DIR.ROOT.joinpath('data/optimize')


def compute_by_h(bc: np.ndarray, hs, order: int):
  H = np.array([[1, 1], [1, 2], [2, 1]], dtype=float)

  K = [data_dir.joinpath('K_o_{}.txt'.format(x + 1)) for x in range(3)]
  Li = [data_dir.joinpath('Lin_o_{}.txt'.format(x + 1)) for x in range(3)]
  Le = [data_dir.joinpath('Lout_o_{}.txt'.format(x + 1)) for x in range(3)]
  C = data_dir.joinpath('C_o.txt')
  Ns = [data_dir.joinpath('specific1_node.txt')]

  system_h = sss.SystemH.from_files(H=H,
                                    C=C,
                                    K=K,
                                    Li=Li,
                                    Le=Le,
                                    Ti=20.0,
                                    Te=5.0,
                                    Ns=Ns)

  thermal_model = tm.ThermalModel(system=system_h)

  dfs = []
  for h in hs:
    ss = thermal_model.state_space(order=order, hi=h[0], he=h[1])
    ys = thermal_model.compute(model=ss, dt=30 * 60, bc=bc, T0=20)

    df = pd.DataFrame(ys, columns=[f'node{x}' for x in range(len(Ns))])
    df['step'] = np.arange(df.shape[0])
    df['hi'] = h[0]
    df['he'] = h[1]

    dfs.append(df)

  dfconcat: pd.DataFrame = pd.concat(dfs, ignore_index=True)

  return dfconcat
