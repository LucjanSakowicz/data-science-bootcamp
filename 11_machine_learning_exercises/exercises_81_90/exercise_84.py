# Wykorzystując funkcję load_digits() z pakietu scikit-learn załadowano dane dotyczące obrazów o rozdzielczości 8x8
# pikseli do zmiennych: data - obrazy zapisane w postaci tablicy numpy o kształcie (1797, 64) target - etykiety,
# cyfry widoczne na obrazach w postaci tablicy numpy o kształcie (1797,) Dokonaj standaryzacji zmiennej data.
# Używając funkcji train_test_split() (ustaw argument random_state=42) podziel dane na zbiór treningowy i testowy:
# X_train, y_train X_test, y_test W odpowiedzi wyświetl kształty otrzymanych tablic tak jak pokazano poniżej.


import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

np.random.seed(42)
data, target = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=42)
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
