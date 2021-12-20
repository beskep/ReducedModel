from typing import Any, List, Optional

# pylint: disable=no-name-in-module
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure
from matplotlib_backend_qtquick.backend_qtquickagg import FigureCanvas
from matplotlib_backend_qtquick.qt_compat import QtCore
from matplotlib_backend_qtquick.qt_compat import QtGui
import numpy as np
import pandas as pd
import seaborn as sns

cmap = sns.color_palette('Dark2')


class PlotController(QtCore.QObject):

  def __init__(self, parent=None) -> None:
    super().__init__(parent)

    self._app: QtGui.QGuiApplication = None
    self._figure: Figure = None
    self._axes: Axes = None
    self._canvas: Optional[FigureCanvas] = None

  def set_app(self, app: QtGui.QGuiApplication):
    self._app = app

  def update_with_canvas(self, canvas: FigureCanvas):
    self._canvas = canvas
    self._figure = self._canvas.figure
    self._axes = self._figure.add_subplot(111)

  def _set_axis(self):
    self._axes.tick_params(axis='both',
                           which='both',
                           top=False,
                           bottom=False,
                           left=False,
                           right=False)

  def clear_plot(self):
    self._axes.clear()
    self._set_axis()

  def draw(self):
    self._canvas.draw()
    self._app.processEvents()


def datetime_format(time, pos):
  """
  float/int 데이터의 x label을 위한 함수.
  label 간격이 적절히 설정되지 않아서 미사용
  """
  h = int(time // 3600)
  d, h = divmod(h, 24)
  m = int((time % 3600) // 60)
  s = int(time % 60)

  return (f'{d} days {h:02d}:{m:02d}:{s:02d}'
          if d else f'{h:02d}:{m:02d}:{s:02d}')


class SimulationPlotController(PlotController):
  TIME0 = np.datetime64(0, 's')

  def __init__(self, parent=None) -> None:
    super().__init__(parent)

    self._app: QtGui.QGuiApplication = None
    self._figure: Figure = None
    self._axes: Axes = None
    self._canvas: Optional[FigureCanvas] = None

    self._models_count = 1
    self._models_names: Optional[List[str]] = None
    self._dt = 0.0  # [sec]
    self._lines = []

  @property
  def models_names(self):
    return self._models_names

  @models_names.setter
  def models_names(self, names: Optional[List[str]]):
    self._models_names = names
    self._models_count = 1 if names is None else len(names)

  @property
  def dt(self):
    return self._dt

  @dt.setter
  def dt(self, value: float):
    self._dt = value

  def update_with_canvas(self, canvas: FigureCanvas):
    super().update_with_canvas(canvas)
    self._set_axis()

  def _set_axis(self):
    super()._set_axis()
    self._axes.set_xlabel('Time step')
    self._axes.set_ylabel('Temperature [ºC]')
    if self.dt:
      self._axes.xaxis.set_major_formatter(DateFormatter('%dd %H:%M:%S'))

  def clear_plot(self):
    super().clear_plot()
    self._lines = []

  def _init_plot(self, points_count: int):
    self.clear_plot()

    points = [f'Point {x+1:02d}' for x in range(points_count)]
    if self._models_count == 1:
      zeros = np.zeros((points_count))
      data: Any = {'Point': points, 'x': zeros, 'y': zeros}
      kwargs = dict(hue='Point')
    else:
      arr = np.array(np.meshgrid(self.models_names, points)).T.reshape([-1, 2])
      data = pd.DataFrame(arr, columns=['Model', 'Point'])
      data[['x', 'y']] = 0.0
      kwargs = dict(hue='Model', style='Point')

    sns.lineplot(data=data, x='x', y='y', ax=self._axes, lw=2.0, **kwargs)
    self._lines = self._axes.get_lines()[:(points_count * self._models_count)]

  def update_plot(self, values: np.ndarray):
    if self._canvas is None:
      raise ValueError

    if not self._lines:
      if values.shape[-1] % self._models_count != 0:
        raise ValueError('모델 개수 설정 오류')
      self._init_plot(points_count=int(values.shape[-1] / self._models_count))

    if values.ndim == 1 or values.shape[0] == 1:
      return

    # set data
    xs: np.ndarray = np.arange(values.shape[0])
    if self.dt:
      # 주의: datetime 데이터를 x축으로 지정하기 때문에,
      # 30일 이상의 데이터의 경우 day가 바르지 않게 표시될 수 있음
      xs = self.TIME0 + (xs * 1000 * self.dt).astype('timedelta64[ms]')

    for idx, line in enumerate(self._lines):
      ys = values[:, idx]
      line.set_xdata(xs)
      line.set_ydata(ys)

    # set lim
    self._axes.set_xlim(xs[0], xs[-1])

    ymin = values.min()
    ymax = values.max()
    if ymin != ymax:
      margin = 0.05 * (ymax - ymin)
      self._axes.set_ylim(ymin - margin, ymax + margin)

    self.draw()


class OptimizationPlotController(PlotController):

  def update_with_canvas(self, canvas: FigureCanvas):
    super().update_with_canvas(canvas)
    self._axes.set_axis_off()

  def _set_axis(self):
    super()._set_axis()
    self._axes.set_xlabel('Model')
    self._axes.set_ylabel('Error [ºC]')

  def plot(self, error: pd.DataFrame, rmse: pd.DataFrame):
    self.clear_plot()

    min_rmse = np.min(rmse['RMSE'])
    rmse['Best'] = ['_nolegend_' if x == min_rmse else '' for x in rmse['RMSE']]

    sns.barplot(data=rmse,
                x='model',
                y='RMSE',
                hue='Best',
                hue_order=['_nolegend_', ''],
                ax=self._axes,
                alpha=0.5,
                dodge=False)

    error['Absolute Error'] = np.abs(error['error'])
    error['Point'] = [f'Point {x}' for x in error['point']]
    error['time'] += np.timedelta64(1, 'D')  # 그래프, 측정치 설정 값에 맞춤
    error['Time'] = error['time'].astype(str)

    sns.scatterplot(data=error,
                    x='model',
                    y='Absolute Error',
                    hue='Time',
                    style='Point',
                    s=100,
                    ax=self._axes)

    if np.any(np.isnan(self._axes.containers[0].datavalues)):
      # 모두 Best Model이 아닐 때 (오차가 모두 같지 않을 때)
      labels = [
          None if np.isnan(x) else 'Best Model'
          for x in self._axes.containers[0].datavalues
      ]
      self._axes.bar_label(self._axes.containers[0],
                           labels=labels,
                           label_type='center',
                           fontsize='large')

    self._axes.legend(bbox_to_anchor=(1, 1))

    self._figure.tight_layout()
    self._set_axis()
    self.draw()
