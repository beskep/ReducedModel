from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import OptimizeResult

from .state_space import SystemH
from .thermal_model import ThermalModel
from .thermal_model import X0Option


@dataclass
class ConstOptions:
  dt: float
  T0: Union[float, None]
  order: Union[int, None]


class ThermalModelOptimizer:

  def __init__(self,
               system: SystemH,
               x0option: Optional[X0Option] = None) -> None:
    self._model = ThermalModel(system=system)

    if x0option is not None:
      self._model.set_x0_option(x0option)

  def predict(self, hi: float, he: float, bc: np.ndarray, opts: ConstOptions):
    ss = self._model.state_space(order=opts.order, hi=hi, he=he)
    pred = self._model.compute(model=ss,
                               dt=opts.dt,
                               bc=bc,
                               T0=opts.T0,
                               progress=False)

    return pred

  def redisual(self, h: np.ndarray, bc: np.ndarray, y: np.ndarray,
               opts: ConstOptions):
    pred = self.predict(hi=h[0], he=h[1], bc=bc, opts=opts)

    yhat = pred[y[:, 0].astype(int)]
    residual = yhat - y[:, 1:]

    return residual.ravel()

  def optimize(
      self,
      bc: np.ndarray,
      y: np.ndarray,
      h0: Union[list, np.ndarray],
      bounds: Tuple,
      opts: ConstOptions,
  ) -> OptimizeResult:
    """
    scipy `least_squares` 함수를 통해 하자부위 실내외 대류열전달계수 최적화

    Parameters
    ----------
    bc : np.ndarray
        Boundary condition. 실내외 온도.

        [[Tinterior, Texterior], ...]
    y : np.ndarray
        검증을 위한 실측 온도 정보

        [[step, y0, y1, ...], ...]

        step은 int (bc의 row index)
        온도를 추출할 node가 하나인 경우 (열화상 평균과 비교하는 일반적 케이스),
        y0 하나만 지정
    h0 : Union[list, np.ndarray]
        초기값. [hi0, he0]
    bounds : Tuple
        탐색 범위. ((hi_min, he_min), (hi_max, he_max))
    opts : ConstOptions
        모델 해석 조건

    Returns
    -------
    OptimizeResult
    """

    # 모델의 대상 노드 수와 입력한 y 개수가 같은지 확인
    if len(self._model.system.Ns) != (y.shape[1] - 1):
      raise ValueError

    max_step = int(np.max(y[:, 0])) + 1
    if bc.shape[0] < max_step:
      raise ValueError

    # 필요한 스텝까지만 자르기
    if bc.shape[0] > max_step:
      bc_ = bc[:max_step]
    else:
      bc_ = bc

    res = least_squares(fun=self.redisual,
                        x0=h0,
                        bounds=bounds,
                        ftol=0.001,
                        xtol=0.001,
                        verbose=2,
                        kwargs=dict(bc=bc_, y=y, opts=opts))

    return res
