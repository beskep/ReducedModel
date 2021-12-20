from functools import wraps
from typing import Optional, Union

# pylint: disable=no-name-in-module
from loguru import logger
from matplotlib_backend_qtquick.qt_compat import QtCore
from matplotlib_backend_qtquick.qt_compat import QtGui

from .plot_controller import OptimizationPlotController
from .plot_controller import SimulationPlotController


def popup(fn):

  @wraps(fn)
  def wrapper(self, *args, **kwargs):
    try:
      res = fn(self, *args, **kwargs)
      if res is None:
        res = True

    except (ValueError, RuntimeError, OSError) as e:
      self.win.show_popup('Error', str(e), 2)

      logger.exception(e)
      res = None

    return res

  return wrapper


class _Window:

  def __init__(self, window: QtGui.QWindow) -> None:
    self._window = window

  def show_popup(self, title: str, message: str, level=0):
    """
    Parameters
    ----------
    title : str
        popup title
    message : str
        popup message
    level : int, optional
        title 왼쪽의 아이콘을 결정.
        0: check, 1: info, 2: warning
    """
    logger.debug('[Popup] {}: {}', title, message)
    self._window.show_popup(title, message, level)
    self.status_message('{}: {}'.format(title, message))

  def progress_bar(self, active: bool):
    self._window.progress_bar(active)

  def status_message(self, message: str):
    self._window.status_message(message)

  def update_model_state(self, model, has_matrix, has_model, has_result):
    self._window.update_model_state(model, has_matrix, has_model, has_result)

  def update_files_list(self, list_):
    self._window.update_files_list(list_)

  def set_points_count(self, count):
    self._window.set_points_count(count)

  def set_best_matching_model(self, model, psi):
    self._window.set_best_matching_model(model, psi)


class BaseController(QtCore.QObject):
  LOGLEVELS = ('TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR',
               'CRITICAL')

  FILE_TYPES = (None, 'Capacitance', 'Conductance', 'Internal Solicitation',
                'External Solicitation', 'Target Nodes')
  SYSTEM_MATRICES_ID = FILE_TYPES[1:-1]  # 각 파일 하나씩 지정
  TARGET_NODES_ID = FILE_TYPES[-1]  # 복수 지정 가능

  OPTION_IDS = ('order', 'deltat', 'initial temperature',
                'internal air temperature', 'external air temperature')

  def __init__(self) -> None:
    super().__init__()

    self._win: Optional[_Window] = None
    self._spc: Optional[SimulationPlotController] = None
    self._opc: Optional[OptimizationPlotController] = None
    self._files = dict()
    self._options = dict()
    self._temperature = dict()  # 실측 온도
    self._points_count = 4  # 시뮬레이션 결과 중 온도 측정 지점 개수

  @property
  def win(self) -> _Window:
    if self._win is None:
      raise ValueError('win not set')

    return self._win

  def set_window(self, win: QtGui.QWindow):
    self._win = _Window(win)

  def set_plot_controller(self, spc: SimulationPlotController,
                          opc: OptimizationPlotController):
    self._spc = spc
    self._opc = opc

  @property
  def points_count(self) -> int:
    return self._points_count

  @points_count.setter
  def points_count(self, value: int):
    self.win.set_points_count(value)
    self._points_count = value

  def _split_level(self, message: str):
    if '|' not in message:
      level = None
    else:
      find = message.find('|')
      level = message[:find].upper()
      message = message[(find + 1):]

    if level not in self.LOGLEVELS:
      level = 'INFO'

    return level, message

  @QtCore.Slot(str)
  def log(self, message: str):
    level, message = self._split_level(message)
    logger.log(level, message)

  @QtCore.Slot(str)
  def select_file_and_type(self, value: str):
    file, index = value.split('|')
    file_type = self.FILE_TYPES[int(index)]
    self._files[file] = file_type

    logger.debug('{}: `{}`', file_type, file)

  @QtCore.Slot(str)
  def delete_file(self, value):
    self._files.pop(value)
    logger.debug('File deleted: `{}`', value)

  @QtCore.Slot()
  def delete_all_files(self):
    self._files = dict()
    logger.debug('Deleted all files')

  @QtCore.Slot(str)
  def set_option(self, value: str):
    v: Union[str, float]
    key, v = value.split('|')

    try:
      v = float(v)
    except ValueError:
      pass

    self._options[key] = v
    logger.debug('Option "{}": {} ({})', key, v, type(v))

  @popup
  def validate_files(self):
    file_ids = list(self._files.values())
    for key in self.SYSTEM_MATRICES_ID:
      if file_ids.count(key) != 1:
        raise ValueError(f'{key} Matrix 설정 오류')

    if not file_ids.count(self.TARGET_NODES_ID):
      raise ValueError('Target Node가 설정되지 않았습니다')

    return True

  @popup
  def validate_options(self):
    for key in self.OPTION_IDS:
      if key not in self._options:
        raise ValueError(f'Option `{key}` 설정 오류')

    return True

  @QtCore.Slot(str, str, str, str, str)
  def temperature_measurement(self, idx, day, time, point, temperature):
    row = int(float(idx)) - 1
    logger.debug('Measurement {}: {} days {} | {} | {} deg', row, day, time,
                 point, temperature)
    self._temperature[row] = (day, time, point, temperature)

  @QtCore.Slot()
  def clear_temperature_measurement(self):
    self._temperature.clear()
