import numpy as np
from loguru import logger
from PyQt5 import QtCore, QtGui

from .plot_controller import PlotController


def popup(fn):

  def wrapper(self, *args, **kwargs):
    try:
      res = fn(self, *args, **kwargs)
      if res is None:
        res = True

    except (ValueError, RuntimeError, OSError) as e:
      self._win.show_popup('Error', str(e), 2)

      logger.exception(e)
      res = None

    return res

  return wrapper


class _Window:

  def __init__(self, window: QtGui.QWindow) -> None:
    self._window = window

  def show_popup(self, title: str, message: str, level=0):
    logger.debug('[Popup] {}: {}', title, message)
    self._window.show_popup(title, message, level)
    self.status_message('{}: {}'.format(title, message))

  def progress_bar(self, active: bool):
    self._window.progress_bar(active)

  def status_message(self, message: str):
    self._window.status_message(message)

  def update_model_state(self, has_matrix, has_model, has_result):
    self._window.update_model_state(has_matrix, has_model, has_result)


class BaseController(QtCore.QObject):
  LOGLEVELS = ('TRACE', 'DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR',
               'CRITICAL')

  FILE_TYPES = (None, 'Damping', 'Stiffness', 'Internal Load', 'External Load',
                'Target Nodes')
  SYSTEM_MATRICES_ID = FILE_TYPES[1:-1]  # 각 파일 하나씩 지정
  TARGET_NODES_ID = FILE_TYPES[-1]  # 복수 지정 가능

  OPTION_IDS = (
      'order',
      'deltat',
      'time steps',
      'initial temperature',
      'internal fluid temperature',
      'internal max temperature',
      'internal min temperature',
      'external fluid temperature',
      'external max temperature',
      'external min temperature',
  )

  QML_PATH_PREFIX = 'file:///'

  def __init__(self) -> None:
    super().__init__()

    self._win: _Window = None
    self._plot_controller: PlotController = None
    self._files = dict()
    self._options = dict()

  def set_window(self, win: QtGui.QWindow):
    self._win = _Window(win)

  def set_plot_controller(self, pc: PlotController):
    self._plot_controller = pc

  def split_level(self, message: str):
    if '|' in message:
      find = message.find('|')
      level = message[:find].upper()
      message = message[(find + 1):]
    else:
      level = None

    if level not in self.LOGLEVELS:
      level = 'INFO'

    return level, message

  @QtCore.Slot(str)
  def log(self, message: str):
    level, message = self.split_level(message)
    logger.log(level, message)

  @QtCore.Slot(str)
  def image_coord(self, value):
    # TODO
    logger.debug(value)

  @QtCore.Slot(str)
  def select_file_and_type(self, value: str):
    file, index = value.split('|')
    index = int(index)
    file_type = self.FILE_TYPES[index]
    self._files[file] = file_type

    logger.debug('{}: {}', file_type, file)

  @QtCore.Slot(str)
  def delete_file(self, value):
    self._files.pop(value)
    logger.debug('File deleted: {}', value)

  @QtCore.Slot()
  def delete_all_files(self):
    self._files = dict()
    logger.debug('Deleted all files')

  @QtCore.Slot(str)
  def set_option(self, value: str):
    key, value = value.split('|')

    try:
      value = float(value)
    except ValueError:
      pass

    self._options[key] = value
    logger.debug('Option {}: {} ({})', key, value, type(value))

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

  @staticmethod
  def sin_temperature_fn(max_temperature, min_temperature, dt):
    if max_temperature <= min_temperature:
      raise ValueError('Tmax <= Tmin')

    avg_temperature = np.average([max_temperature, min_temperature])
    amplitude = (max_temperature - min_temperature) / 2.0
    k = 2 * np.pi / (24 * 3600 / dt)

    def fn(step):
      return avg_temperature + amplitude * np.sin(k * step)

    return fn
