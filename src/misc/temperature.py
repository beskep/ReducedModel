from pathlib import Path
from typing import Union

import pandas as pd


def read_temperature(path: Union[str, Path]) -> pd.DataFrame:
  df = pd.read_csv(path)
  df.columns = ['Time', 'Temperature', 'RelativeHumidity']

  return df
