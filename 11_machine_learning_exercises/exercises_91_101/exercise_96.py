# Skopiowano obiekt df do zmiennej data. Następnie wyrwano kolumnę target ze zmiennej data i przypisano do zmiennej
# target. Wykorzystując funkcję train_test_split() podziel dane (data, target) na zbiór treningowy i testowy (użyj
# argumentu random_datate=42) i przypisz odpowiednio do zmiennych: data_train, target_train data_test, target_test W
# odpowiedzi wyświetl kształty obiektów: data_train, target_train, data_test, target_test tak jak pokazano poniżej.


import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target'],
)

data = df.copy()
target = data.pop('target')

data_train, data_test, target_train, target_test = train_test_split(data, target, random_state=42)
print(f'data_train shape: {data_train.shape}')
print(f'target_train shape: {target_train.shape}')
print(f'data_test shape: {data_test.shape}')
print(f'target_test shape: {target_test.shape}')
