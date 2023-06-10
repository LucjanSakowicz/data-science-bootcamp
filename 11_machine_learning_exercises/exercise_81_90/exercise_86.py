# Wykorzystując funkcję load_digits() z pakietu scikit-learn załadowano dane dotyczące obrazów o rozdzielczości 8x8
# pikseli do zmiennych: data - obrazy zapisane w postaci tablicy numpy o kształcie (1797, 64) target - etykiety,
# cyfry widoczne na obrazach w postaci tablicy numpy o kształcie (1797,) Następnie dokonano standaryzacji zmiennej
# data. Używając funkcji train_test_split() podzielono dane na zbiór treningowy i testowy: X_train, y_train X_test,
# y_test o kształtach odpowiednio: X_train shape: (1347, 64) y_train shape: (1347,) X_test shape: (450, 64) y_test
# shape: (450,) Wykorzystując klasę LogisticRegression z pakietu scikit-learn zbuduj model klasyfikacji. Wyucz model
# na danych treningowych i następnie dokonaj oceny na danych testowych. Dokładność modelu wyświetl do konsoli tak jak
# pokazano poniżej.

import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


np.random.seed(42)
data, target = load_digits(return_X_y=True)
data = data / data.max()

X_train, X_test, y_train, y_test = train_test_split(
    data, target, random_state=42
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
print(f'Logistic Regression accuracy: {model.score(X_test, y_test):.4f}')