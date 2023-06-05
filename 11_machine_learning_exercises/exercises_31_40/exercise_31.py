# Wykorzystując funkcję train_test_split() z pakietu scikit-learn podziel dane (data oraz target) na zbiór treningowy
# i testowy, odpowiednio: zbiór treningowy: X_train, y_train zbiór testowy: X_test, y_test Ustaw argument
# random_state=40 oraz rozmiar zbioru testowego na 25%. W odpowiedzi wydrukuj rozmiary tablic: X_train, y_train,
# X_test, y_test tak jak pokazano poniżej.

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

print(f'X_train shape {X_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'X_test shape {X_test.shape}')
print(f'y_test shape {y_test.shape}')