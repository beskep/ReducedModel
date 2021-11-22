from itertools import cycle
from itertools import product
from typing import Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib_backend_qtquick.backend_qtquickagg import FigureCanvas
from matplotlib_backend_qtquick.qt_compat import QtCore
from matplotlib_backend_qtquick.qt_compat import QtGui
import numpy as np

try:
  import seaborn as sns

  cmap = sns.color_palette('Dark2')
except ImportError:
  from matplotlib.cm import get_cmap

  cmap = get_cmap('Dark2').colors


class PlotController(QtCore.QObject):

  def __init__(self, parent=None) -> None:
    super().__init__(parent)

    self._app: QtGui.QGuiApplication = None
    self._figure: Figure = None
    self._axes: Axes = None
    self._canvas: Optional[FigureCanvas] = None

    self._lines = []

  def set_app(self, app: QtGui.QGuiApplication):
    self._app = app

  def update_with_canvas(self, canvas: FigureCanvas):
    self._canvas = canvas
    self._figure = self._canvas.figure
    self._axes = self._figure.add_subplot(111)
    self._set_axis()

  def _set_axis(self):
    self._axes.set_xlabel('Time step')
    self._axes.set_ylabel('Temperature [ºC]')
    self._axes.tick_params(axis='both',
                           which='both',
                           top=False,
                           bottom=False,
                           left=False,
                           right=False)

  def clear_plot(self):
    self._axes.clear()
    self._lines = []
    self._set_axis()

  @staticmethod
  def styles():
    return cycle(product(['-', '--', ':'], cmap))

  def _init_plot(self, line_names: list):
    self.clear_plot()

    styles = self.styles()
    for _ in range(len(line_names)):
      ls, color = next(styles)
      line = self._axes.plot([0], [0], color=color, linestyle=ls)
      self._lines.append(line[0])

    self._axes.legend(self._lines, line_names)

  def update_plot(self, values: np.ndarray):
    if self._canvas is None:
      raise ValueError

    if not self._lines:
      lines_count = values.shape[1]
      self._init_plot(line_names=[f'loc {x+1}' for x in range(lines_count)])

    # set data
    xs = np.arange(values.shape[0])
    for idx, line in enumerate(self._lines):
      ys = values[:, idx]
      line.set_xdata(xs)
      line.set_ydata(ys)

    # set xlim
    self._axes.set_xlim(0, values.shape[0])

    # set ylim
    ymin = values.min()
    ymax = values.max()
    margin = 0.05 * (ymax - ymin)
    self._axes.set_ylim(ymin - margin, ymax + margin)
    self._axes.set_ymargin(0.1)

    # draw
    self._canvas.draw()
    self._app.processEvents()
