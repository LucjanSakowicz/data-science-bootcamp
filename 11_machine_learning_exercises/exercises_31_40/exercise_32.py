# Podane są poniższe tablice: data target zbiór treningowy: X_train, y_train zbiór testowy: X_test, y_test Sprawdź
# procentowy rozkład wartości zmiennych target, y_train oraz y_test. Wynik wydrukuj do konsoli tak jak pokazano
# poniżej.

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 200)
np.set_printoptions(precision=2, suppress=True, linewidth=100)
raw_data = load_breast_cancer()

data = raw_data['data']
target = raw_data['target']

X_train, X_test, y_train, y_test = train_test_split(
    data, target, random_state=40, test_size=0.25
)

print(f'target :{np.unique(target, return_counts=True)[1] / len(target)}')
print(f'y_train:{np.unique(y_train, return_counts=True)[1] / len(y_train)}')
print(f'y_test :{np.unique(y_test, return_counts=True)[1] / len(y_test)}')
