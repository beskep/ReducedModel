import multiprocessing as mp
from typing import Optional

from loguru import logger
import numpy as np
from PyQt5 import QtCore
import reduced_model as rm
from temperature import read_temperature

from rm import utils

from .base_controller import BaseController
from .base_controller import popup


def model_reduction_producer(queue: mp.Queue, order, mtxA, mtxB, mtxJ):
  utils.set_logger()
  system, order = rm.reduce_model(order, mtxA, mtxB, mtxJ)
  queue.put((system, order))


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

    self._reducer = rm.ModelReducer(order=0)
    self._results = None

  @popup
  def _read_matrices(self):
    self._reducer.order = self._options['order']

    # 해석 환경 설정
    id_file = {
        value: key
        for key, value in self._files.items()
        if value != self.TARGET_NODES_ID
    }
    self._reducer.read_matrices(*[id_file[x] for x in self.SYSTEM_MATRICES_ID])

    self._reducer.set_target_nodes([
        key for key, value in self._files.items()
        if value == self.TARGET_NODES_ID
    ])

    self._reducer.set_fluid_temperature(
        interior=self._options['internal air temperature'],
        exterior=self._options['external air temperature'])

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

    mtxA, mtxB, mtxJ = self._reducer.compute_state_matrices()
    queue = mp.Queue()

    self._consumer.set_queue(queue)
    self._consumer.start()

    logger.info('Model reducing start')
    process = mp.Process(name='model reduce',
                         target=model_reduction_producer,
                         args=(queue, self._options['order'], mtxA, mtxB, mtxJ),
                         daemon=True)
    process.start()

  @QtCore.Slot()
  def reduce_model(self):
    self._win.progress_bar(True)
    self._win.show_popup(
        title='모델 리듀스 시작',
        message='환경에 따라 1분 이상 걸릴 수 있습니다.',
        level=1,
    )
    self._reduce_model()

  @popup
  def reduce_model_done(self):
    rsystem, rorder = self._consumer.get_result()
    self._reducer.set_reduced_model(rsystem, rorder)

    logger.info('Model reducing ended')
    self._win.progress_bar(False)
    self._win.show_popup('Success', '모델 리듀스 완료.')

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
      self._win.show_popup('Success', '모델 로드 완료.')

  @popup
  def _compute(self, dt, time_steps, initial_temperature=0.0):
    has_reduced_model = self._reducer.has_reduced_model()
    if not has_reduced_model:
      self._win.show_popup(
          title='Warning',
          message='모델이 리듀스 되지 않았습니다. 원본 모델을 통해 계산합니다.',
          level=1,
      )
    res = self._reducer.compute(dt=dt,
                                time_step=time_steps,
                                initial_temperature=initial_temperature,
                                callback=self._plot_controller.update_plot,
                                reduced_system=has_reduced_model)
    self._results = res

    return True

  @QtCore.Slot()
  def compute(self):
    dt = self._options['deltat']
    time_steps = int(self._options['time steps'])
    initial_temperature = self._options['initial temperature']

    interior_temperature_fn = self.sin_temperature_fn(
        max_temperature=self._options['internal max temperature'],
        min_temperature=self._options['internal min temperature'],
        dt=dt)
    external_temperature_fn = self.sin_temperature_fn(
        max_temperature=self._options['external max temperature'],
        min_temperature=self._options['external min temperature'],
        dt=dt)

    self._reducer.set_temperature_condition(fn=interior_temperature_fn,
                                            loc=rm.Location.Interior)
    self._reducer.set_temperature_condition(fn=external_temperature_fn,
                                            loc=rm.Location.Exterior)

    self._plot_controller.clear_plot()
    if self._compute(dt, time_steps, initial_temperature):
      self._win.show_popup('Success', '연산 완료.')
      self.update_model_state()

  @popup
  def _save_model(self, path: str):
    try:
      self._reducer.save_reduced_model(path)
    except ValueError as e:
      raise ValueError('모델이 존재하지 않음.') from e

    return True

  @QtCore.Slot(str)
  def save_model(self, path: str):
    path = self.to_valid_path(path)

    if self._save_model(path):
      self.win.show_popup('Success', '모델 저장 완료.')

  def read_model(self, path):
    path = self.to_valid_path(path)
    self._reducer.load_reduced_model(path)

    self._reducer.set_fluid_temperature(
        interior=self._options['internal air temperature'],
        exterior=self._options['external air temperature'])

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
    has_matrix = self._reducer.has_matrices()
    has_model = self._reducer.has_reduced_model()
    has_result = self._results is not None

    self._win.update_model_state(has_matrix, has_model, has_result)
