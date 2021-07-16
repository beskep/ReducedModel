import context
import utils

import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype

from temperature import read_temperature

DATA_DIR = utils.ROOT_DIR.joinpath('data')


def test_read_csv():
  path = DATA_DIR.joinpath('temperature_example.csv')
  assert path.exists()

  df = read_temperature(path=path)

  assert df.dtypes[0] == pd.Timestamp
  assert is_numeric_dtype(df['Temperature'])
  assert is_numeric_dtype(df['RelativeHumidity'])


if __name__ == '__main__':
  pytest.main(['-vv', '-k', 'test_temperature'])
