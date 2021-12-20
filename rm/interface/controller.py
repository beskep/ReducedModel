import multiprocessing as mp
from typing import Optional

# pylint: disable=no-name-in-module
from loguru import logger
from matplotlib_backend_qtquick.qt_compat import QtCore
import numpy as np
import pandas as pd

from rm import utils
from rm.interface.reference_systems import ReferenceSystems
from rm.reduced_model.state_space import reduce_model
from rm.reduced_model.state_space import StateSpace
from rm.reduced_model.state_space import System
from rm.reduced_model.thermal_model import ThermalModel

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


def valid_path(path: str):
  return path.replace('file:///', '')


def _temperature_error(measured: pd.DataFrame, simulated: pd.DataFrame):
  measured_times = np.unique(measured['time'])
  simulation_range = (np.min(simulated['time'].values),
                      np.max(simulated['time'].values))
  for time in measured_times:
    if not (simulation_range[0] <= time <= simulation_range[1]):
      pdtd = pd.Timedelta(time)
      raise ValueError(f'시뮬레이션 범위에 포함되지 않는 시간이 입력됐습니다: {pdtd}')

  # FIXME 비효율적
  models = np.unique(simulated['model'])
  points = np.unique(measured['point'])
  dfs = []
  for point in points:
    for model in models:
      dfpm = simulated.loc[(simulated['point'] == point) &
                           (simulated['model'] == model), :]
      temperature = np.interp(measured_times.astype(float),
                              xp=dfpm['time'].values.astype(float),
                              fp=dfpm['value'].values)
      dfs.append(
          pd.DataFrame({
              'time': measured_times,
              'point': point,
              'model': model,
              'simulated': temperature,
          }))

  error: pd.DataFrame = pd.concat(dfs)
  error = pd.merge(left=error, right=measured, on=['time', 'point'])
  error['error'] = error['measurement'] - error['simulated']

  rmse: pd.DataFrame = error.groupby('model')['error'].apply(
      lambda x: np.sqrt(np.mean(np.square(x)))).to_frame().reset_index()
  rmse = rmse.rename(columns={'error': 'RMSE'})

  return error, rmse


class Controller(BaseController):

  def __init__(self) -> None:
    super().__init__()

    self._consumer = ModelReductionConsumer()
    self._consumer.done.connect(self.reduce_model_done)

    self._reduced_system: Optional[StateSpace] = None
    self._reference_systems: Optional[ReferenceSystems] = None
    self._thermal_model: Optional[ThermalModel] = None
    self._results: Optional[pd.DataFrame] = None

  def reset_model(self):
    self._reduced_system = None
    self._thermal_model = None
    self._reference_systems = None
    self._spc.models_names = None
    self.clear_results()

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
    air_temperature = [
        self._options[f'{x} air temperature'] + 273.15
        for x in ['internal', 'external']
    ]

    system = System.from_files(C=id_file['Capacitance'],
                               K=id_file['Conductance'],
                               Li=id_file['Internal Solicitation'],
                               Le=id_file['External Solicitation'],
                               Ti=air_temperature[0],
                               Te=air_temperature[1],
                               Ns=target_nodes)
    self._thermal_model = ThermalModel(system=system)
    self.points_count = len(target_nodes)

    logger.info('Read matrices')

    return True

  @QtCore.Slot()
  def read_matrices(self):
    if not self.validate_files():
      return
    if not self.validate_options():
      return

    self.reset_model()

    if self._read_matrices():
      self.win.show_popup(title='Success', message='행렬 로드 완료.')

    self.update_model_state()

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
    if self._reference_systems is not None:
      self.win.show_popup(title='Information',
                          message='레퍼런스 모델은 이미 리듀스 되어 있습니다.',
                          level=1)
      return

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
    self.win.show_popup(title='Success', message='모델 리듀스 완료.')

    self.clear_results()
    self.update_model_state()

  @popup
  def _read_user_selected_model(self):
    if len(self._files) == 0:
      raise FileNotFoundError('모델 파일을 선택해주세요.')

    if len(self._files) > 1:
      raise ValueError('모델 파일을 둘 이상 선택했습니다.')

    return self.read_model(list(self._files.keys())[0])

  @QtCore.Slot()
  def read_user_selected_model(self):
    self.reset_model()

    if self._read_user_selected_model():
      self.clear_results()
      self.win.show_popup(title='Success', message='모델 로드 완료.')

    self.clear_results()
    self.update_model_state()

  @QtCore.Slot()
  def read_reference_models(self):
    self.reset_model()

    if self._reference_systems is None:
      self._reference_systems = ReferenceSystems()
    self._spc.models_names = self._reference_systems.names

    self.win.show_popup(title='Success', message='레퍼런스 모델 로드 완료.')

    self.clear_results()
    self.update_model_state()

  @QtCore.Slot()
  def update_reference_models(self):
    if self._reference_systems is None:
      self._reference_systems = ReferenceSystems()

    names = self._reference_systems.names
    self.win.update_files_list(list(names))
    self.points_count = 4

  @popup
  def _compute(self, dt: float, bc: np.ndarray, T0=0.0):
    flag_reference = False
    if self._reference_systems is not None:
      model = self._reference_systems.load()
      flag_reference = True
    elif self._reduced_system is not None:
      model = self._reduced_system
    else:
      if self._thermal_model is None:
        raise ValueError('행렬이 로드되지 않았습니다.')

      self.win.show_popup(
          title='Warning',
          message='모델이 리듀스 되지 않았습니다. 원본 모델을 통해 계산합니다.',
          level=1,
      )
      model = self._thermal_model.state_space(order=None)

    assert self._spc is not None
    self._spc.dt = dt
    if self._thermal_model is None:
      self._thermal_model = ThermalModel(None)

    kwargs = dict(dt=dt, bc=bc, T0=T0, callback=self._spc.update_plot)
    if flag_reference:
      res = self._thermal_model.compute_multi_systems(sss=model, **kwargs)
      columns = [
          f'{m}:Point{p+1:02d}' for m in self._reference_systems.names
          for p in range(self.points_count)
      ]
    else:
      res = self._thermal_model.compute(ss=model, **kwargs)
      columns = [f'Point{p+1:02d}' for p in range(self.points_count)]

    df = pd.DataFrame(res, columns=columns)

    time = (np.arange(bc.shape[0]) * 1000 * dt).astype('timedelta64[ms]')
    df.insert(0, column='time', value=time)
    self._results = df

    return True

  @popup
  def _read_compute_options(self):
    try:
      dt = self._options['deltat']
      T0 = self._options['initial temperature']
      temperature_path = self._options['temperature log path']
    except KeyError as e:
      raise ValueError(f'설정되지 않은 옵션: {e}') from e

    temperature = pd.read_csv(temperature_path)

    return dt, T0, temperature.values

  @QtCore.Slot()
  def compute(self):
    options = self._read_compute_options()
    if options is None:
      return

    self._spc.clear_plot()
    if self._compute(dt=options[0], bc=options[2], T0=options[1]):
      self._win.show_popup('Success', '연산 완료.')
      self.update_model_state()

  @popup
  def _save_model(self, path: str):
    if self._reference_systems is not None:
      self.win.show_popup(title='Information',
                          message='레퍼런스 모델은 저장할 수 없습니다.',
                          level=1)
      return False

    if self._thermal_model is None:
      if self._reduced_system is None:
        raise ValueError('저장할 모델이 없습니다.')
      self._thermal_model = ThermalModel(None)

    self._thermal_model.save(path=path, state_space=self._reduced_system)

    return True

  @QtCore.Slot(str)
  def save_model(self, path: str):
    path = valid_path(path)

    if self._save_model(path):
      self.win.show_popup('Success', '모델 저장 완료.')

  def read_model(self, path):
    path = valid_path(path)
    self._reduced_system = ThermalModel.load(path)

    return True

  @popup
  def _save_results(self, path):
    if self._results is None:
      raise ValueError('시뮬레이션 결과가 존재하지 않습니다.')

    self._results.to_csv(path, index=False)

    return True

  @QtCore.Slot(str)
  def save_results(self, path: str):
    path = valid_path(path)
    if self._save_results(path):
      self.win.show_popup('Success', '결과 저장 완료.')

  def clear_results(self):
    self._results = None

    if self._spc is None:
      raise RuntimeError('PlotController가 설정되지 않음')
    self._spc.clear_plot()

  def update_model_state(self):
    has_matrix = (self._thermal_model is not None or
                  self._reference_systems is not None)
    has_model = (self._reduced_system is not None or
                 self._reference_systems is not None)
    has_result = self._results is not None

    if self._reference_systems is not None:
      model = 'reference'
    elif has_matrix or has_model:
      model = 'matrix'
    else:
      model = 'none'

    self._win.update_model_state(model, has_matrix, has_model, has_result)

  def _prep_temperature_measurement(self) -> pd.DataFrame:
    TIME0 = '2000-01-01'

    try:
      data = [self._temperature[x] for x in range(len(self._temperature))]
    except KeyError as e:
      raise ValueError(f'온도 측정값 설정 오류 (row {e})') from e

    df = pd.DataFrame(data=data,
                      columns=['_day', '_time', 'point', 'measurement'])
    df['measurement'] = df['measurement'].astype(float)
    df['point'] = df['point'].str.replace('Point ', '').astype(int)

    df['time'] = pd.to_datetime(TIME0 + ' ' + df['_time'])
    # GUI의 입력치가 1일부터 시작하기 때문에 `_day`에서 1을 빼줌
    df['time'] += pd.to_timedelta(df['_day'].astype(int) - 1, unit='D')

    # 각 측정값에 동일한 시간 적용
    df.loc[(np.arange(df.shape[0]) % self.points_count != 0), 'time'] = pd.NaT
    df['time'] = df['time'].fillna(method='ffill') - np.datetime64(TIME0, 'ms')

    return df[['time', 'point', 'measurement']]

  def _prep_simulation_result(self) -> pd.DataFrame:
    if self._results is None:
      raise ValueError('시뮬레이션 결과가 존재하지 않습니다.')

    df = pd.melt(self._results, id_vars='time')
    df[['model', 'point']] = df['variable'].str.split(':', expand=True)
    df['point'] = df['point'].str.replace('Point', '').astype(int)

    return df

  @popup
  def _optimize(self):
    assert self._opc is not None
    if self._reference_systems is None:
      raise ValueError('레퍼런스 모델의 시뮬레이션 결과만 최적화 가능합니다.')

    simulated = self._prep_simulation_result()
    measured = self._prep_temperature_measurement()
    error, rmse = _temperature_error(measured=measured, simulated=simulated)

    self._opc.plot(error=error, rmse=rmse)

    model = rmse.loc[rmse['RMSE'] == np.min(rmse['RMSE']), 'model'].values[0]
    psi = self._reference_systems.linear_thermal_transmittance[model]
    self.win.set_best_matching_model(model, psi)  # TODO formatting

  @QtCore.Slot()
  def optimize(self):
    self._optimize()
