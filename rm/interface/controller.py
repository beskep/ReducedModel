import multiprocessing as mp
from typing import Optional

from loguru import logger
from matplotlib_backend_qtquick.qt_compat import QtCore
import numpy as np

from rm import utils
from rm.reduced_model.state_space import reduce_model
from rm.reduced_model.state_space import StateSpace
from rm.reduced_model.state_space import System
from rm.reduced_model.thermal_model import ThermalModel
from rm.temperature import read_temperature

from .base_controller import BaseController
from .base_controller import popup


def reduce_producer(queue: mp.Queue, state_space: StateSpace, order: int):
  utils.set_logger()
  reduced = reduce_model(state_space=state_space, order=order)
  queue.put(reduced)


class ModelReductionConsumer(QtCore.QThread):
  done = QtCore.Signal()

  def __init__(self, parent=None) -> None:
    super().__init__(parent=parent)

    self._queue: Optional[mp.Queue] = None
    self._res = None

  def set_queue(self, queue: mp.Queue):
    self._queue = queue

  def get_result(self):
    return self._res

  def run(self) -> None:
    if self._queue is None:
      raise ValueError

    while True:
      if not self._queue.empty():
        self._res = self._queue.get()
        self.done.emit()
        break


class Controller(BaseController):

  def __init__(self) -> None:
    super().__init__()

    self._consumer = ModelReductionConsumer()
    self._consumer.done.connect(self.reduce_model_done)

    self._reduced_system: Optional[StateSpace] = None
    self._thermal_model: Optional[ThermalModel] = None
    self._results = None

  @popup
  def _read_matrices(self):
    target_nodes = [
        k for k, v in self._files.items() if v == self.TARGET_NODES_ID
    ]
    id_file = {
        value: key
        for key, value in self._files.items()
        if value != self.TARGET_NODES_ID
    }

    system = System.from_files(C=id_file['Capacitance'],
                               K=id_file['Conductance'],
                               Li=id_file['Internal Solicitation'],
                               Le=id_file['External Solicitation'],
                               Ti=self._options['internal air temperature'],
                               Te=self._options['external air temperature'],
                               Ns=target_nodes)

    self._thermal_model = ThermalModel(system=system)
    self._reduced_system = None

    logger.info('Read matrices')

    return True

  @QtCore.Slot()
  def read_matrices(self):
    if not self.validate_files():
      return
    if not self.validate_options():
      return

    if self._read_matrices():
      self._win.show_popup('Success', '행렬 로드 완료.')

    self.update_model_state()

  @QtCore.Slot(str)
  def read_temperature(self, message: str):
    loc, path = message.split('|')

    try:
      df = read_temperature(path=path)
    except (ValueError, OSError, RuntimeError):
      logger.exception('온도 파일 로드 실패')
      return

    print(df)

  @popup
  def _reduce_model(self):
    ss = self._thermal_model.state_space(order=None)

    queue = mp.Queue()
    self._consumer.set_queue(queue)
    self._consumer.start()

    logger.info('Model reducing start')
    process = mp.Process(name='reduce',
                         target=reduce_producer,
                         args=(queue, ss, self._options['order']),
                         daemon=True)
    process.start()

  @QtCore.Slot()
  def reduce_model(self):
    if self._reduced_system is not None:
      self.win.show_popup(title='Information',
                          message='이미 리듀스된 모델이 있습니다.',
                          level=1)
      return
    if self._thermal_model is None:
      self.win.show_popup(
          title='Error',
          message='행렬이 로드되지 않았습니다.',
          level=2,
      )
      return

    self.win.progress_bar(True)
    self.win.show_popup(title='모델 리듀스 시작',
                        message='환경에 따라 1분 이상 걸릴 수 있습니다.',
                        level=1)
    self._reduce_model()

  @popup
  def reduce_model_done(self):
    self._reduced_system = self._consumer.get_result()

    logger.info('Model reducing ended')
    self.win.progress_bar(False)
    self.win.show_popup('Success', '모델 리듀스 완료.')

    self.clear_results()

  def to_valid_path(self, path: str):
    return path.replace(self.QML_PATH_PREFIX, '')

  @popup
  def _read_model_from_selected(self):
    if len(self._files) == 0:
      raise FileNotFoundError('모델 파일을 선택해주세요.')

    if len(self._files) > 1:
      raise ValueError('모델 파일을 둘 이상 선택했습니다.')

    return self.read_model(list(self._files.keys())[0])

  @QtCore.Slot()
  def read_model_from_selected(self):
    if self._read_model_from_selected():
      self.clear_results()
      self.win.show_popup('Success', '모델 로드 완료.')

  @popup
  def _compute(self, dt, bc, T0=0.0):
    if self._reduced_system is None:
      if self._thermal_model is None:
        raise ValueError('행렬이 로드되지 않았습니다.')

      self.win.show_popup(
          title='Warning',
          message='모델이 리듀스 되지 않았습니다. 원본 모델을 통해 계산합니다.',
          level=1,
      )
      ss = self._thermal_model.state_space(order=None)
    else:
      ss = self._reduced_system

    res = self._thermal_model.compute(
        ss=ss,
        dt=dt,
        bc=bc,
        T0=T0,
        callback=self._plot_controller.update_plot,
    )
    self._results = res

    return True

  @QtCore.Slot()
  def compute(self):
    dt = self._options['deltat']
    time_steps = int(self._options['time steps'])
    T0 = self._options['initial temperature']

    # FIXME csv로 읽은 온도로
    interior_temperature_fn = self.sin_temperature_fn(
        max_temperature=self._options['internal max temperature'],
        min_temperature=self._options['internal min temperature'],
        dt=dt)
    external_temperature_fn = self.sin_temperature_fn(
        max_temperature=self._options['external max temperature'],
        min_temperature=self._options['external min temperature'],
        dt=dt)
    Ti = [interior_temperature_fn(x) for x in range(time_steps)]
    To = [external_temperature_fn(x) for x in range(time_steps)]
    bc = np.vstack((Ti, To)).T

    self._plot_controller.clear_plot()
    if self._compute(dt=dt, bc=bc, T0=T0):
      self._win.show_popup('Success', '연산 완료.')
      self.update_model_state()

  @popup
  def _save_model(self, path: str):
    self._thermal_model.save(path=path, state_space=self._reduced_system)

    return True

  @QtCore.Slot(str)
  def save_model(self, path: str):
    path = self.to_valid_path(path)

    if self._save_model(path):
      self.win.show_popup('Success', '모델 저장 완료.')

  def read_model(self, path):
    path = self.to_valid_path(path)
    self._reduced_system = ThermalModel.load(path)
    self._thermal_model = ThermalModel(None)

    return True

  @popup
  def _save_results(self, path):
    if self._results is None:
      raise ValueError('결과가 존재하지 않습니다.')

    np.savetxt(fname=path, X=self._results, fmt='%.9e', delimiter=',')

    return True

  @QtCore.Slot(str)
  def save_results(self, path: str):
    path = self.to_valid_path(path)
    if self._save_results(path):
      self.win.show_popup('Success', '결과 저장 완료.')

  def clear_results(self):
    self._results = None

    if self._plot_controller is None:
      raise RuntimeError('PlotController가 설정되지 않음')
    self._plot_controller.clear_plot()

    self.update_model_state()

  def update_model_state(self):
    has_matrix = self._thermal_model is not None
    has_model = self._reduced_system is not None
    has_result = self._results is not None

    self._win.update_model_state(has_matrix, has_model, has_result)
