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


@click.command()
@click.option('--loglevel',
              '-l',
              default='INFO',
              help='로그 표시 레벨 (debug, info, ...)')
@click.option('--output', '-o', help='결과 파일 저장 경로 (csv)')
@click.argument('config_path')
def cli(config_path, loglevel, output):
  _set_logger(loglevel)

  config_path = Path(config_path).resolve()
  config_path.stat()

  with config_path.open('r') as f:
    config = yaml.safe_load(f)

  env = config['environment']
  paths = _config_paths(config=config, config_dir=config_path.parent)
  temperature = _read_temperature(path=env['temperature_path'],
                                  config_dir=config_path.parent)

  system = System.from_files(C=paths[0],
                             K=paths[1],
                             Li=paths[2],
                             Le=paths[3],
                             Ti=config['model']['air_temperature']['internal'],
                             Te=config['model']['air_temperature']['external'],
                             Ns=paths[4:])

  order = env['order']
  if order is None:
    logger.info('모델 리덕션을 시행하지 않음')
  else:
    order = int(order)
    logger.debug('모델 리덕션 목표 차수: {}', order)

  model = ThermalModel(system=system)
  ss = model.state_space(order=order)
  logger.debug('모델 차수: {}', ss.A.shape[0])

  out = model.compute(ss=ss,
                      dt=float(env['dt']),
                      bc=temperature,
                      T0=float(env['initial_temperature']),
                      progress=True)
  df = pd.DataFrame(out)

  if output is None:
    logger.info('시뮬레이션 완료:')
    print(df)
  else:
    logger.info('시뮬레이션 결과 저장: "{}"', output)
    df.to_csv(output)


if __name__ == '__main__':
  cli()