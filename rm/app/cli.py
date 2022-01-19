from pathlib import Path

import click
from loguru import logger
import pandas as pd
import yaml

from rm.reduced_model.state_space import System
from rm.reduced_model.thermal_model import ThermalModel
from rm.utils import set_logger

_KEYS = (
    'capacitance',
    'conductance',
    ('solicitation', 'internal'),
    ('solicitation', 'external'),
    'target_nodes',
)


def _get_path(paths: dict, key):
  if isinstance(key, str):
    return paths[key]

  return paths[key[0]][key[1]]


def _valid_path(key, path, config_dir: Path):
  path = Path(path)
  ks = key if isinstance(key, str) else f'{key[0]}/{key[1]}'

  if not path.exists():
    path = config_dir.joinpath(path)
    if not path.exists():
      raise FileNotFoundError(f'{ks} 파일을 찾지 못했습니다: {path}')

  logger.debug('{} matrix path: "{}"', ks.title(), path)

  return path


def _config_paths(config: dict, config_dir: Path):
  paths = config['model']['matrix_path']
  ps = [_get_path(paths, x) for x in _KEYS[:-1]] + paths[_KEYS[-1]]
  ks = list(_KEYS[:-1]) + [_KEYS[-1]] * len(paths[_KEYS[-1]])

  return [_valid_path(k, p, config_dir) for k, p in zip(ks, ps)]


def _read_temperature(path, config_dir: Path):
  path = Path(path)
  if not path.exists():
    path = config_dir.joinpath(path)

  if not path.exists():
    raise FileNotFoundError(f'온도 파일을 찾지 못했습니다: {path}')

  logger.debug('Temperature path: "{}"', path)

  return pd.read_csv(path).values


def _set_logger(loglevel):
  try:
    l = int(loglevel)
  except (ValueError, TypeError):
    l = loglevel

  set_logger(l)


def _system(config, config_dir: Path):
  paths = _config_paths(config=config, config_dir=config_dir)
  air_temperature = config['model']['air_temperature']

  # State-Space System
  system = System.from_files(C=paths[0],
                             K=paths[1],
                             Li=paths[2],
                             Le=paths[3],
                             Ti=air_temperature['internal'],
                             Te=air_temperature['external'],
                             Ns=paths[4:])

  return system


def main(config_path, output=None):
  config_path = Path(config_path).resolve()
  config_path.stat()
  logger.info('config: "{}"', config_path)

  if output is not None:
    output = Path(output)
    if not output.is_dir():
      raise NotADirectoryError(output)
    output.stat()
  logger.info('output: "{}"', output)

  with config_path.open('r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

  env = config['environment']
  temperature = _read_temperature(path=env['temperature_path'],
                                  config_dir=config_path.parent)

  system = _system(config=config, config_dir=config_path.parent)
  logger.info('모델 차수: {}', system.C.shape[0])

  order = env['order']
  if order is None:
    logger.info('모델 리덕션을 시행하지 않음')
  else:
    order = int(order)
    logger.info('모델 리덕션 목표 차수: {}', order)

  model = ThermalModel(system=system)
  method = env['reduction_method']
  method = None if not method else str(method).lower()
  ss = model.state_space(order=order, reduction_method=method)
  if order:
    logger.info('리덕션 후 모델 차수: {}', ss.A.shape[0])

  T0 = env['initial_temperature']
  out = model.compute(ss=ss,
                      dt=float(env['dt']),
                      bc=temperature,
                      T0=(None if T0 is None else float(T0)),
                      progress=True)
  df = pd.DataFrame(out)

  if output is None:
    logger.info('시뮬레이션 완료:')
    print(df)
  else:
    df.to_csv(output.joinpath('SimulatedTemperature.csv'))
    model.save(path=output.joinpath('Model.npz').as_posix(), state_space=ss)
    logger.info('시뮬레이션 결과 저장 완료: "{}"', output)

  return df


@click.command()
@click.option('--loglevel',
              '-l',
              default='INFO',
              help='로그 표시 레벨 (debug, info, ...)')
@click.argument('config_path')
@click.argument('output', required=False)
def cli(loglevel, config_path, output):  # pylint: disable-all
  """
  수치모델 축소 및 시뮬레이션

  \b
  Arguments:
      config_path: config.yaml 경로
      output:      결과 저장 경로 (폴더)
  """
  _set_logger(loglevel)
  main(config_path=config_path, output=output)


if __name__ == '__main__':
  # pylint: disable=no-value-for-parameter
  cli()
