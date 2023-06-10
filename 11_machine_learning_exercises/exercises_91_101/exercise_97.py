# Przygotowano poniższe zbiory: data_train, target_train data_test, target_test Wykorzystując klasę LinearRegression
# (z domyślnymi parametrami) z pakietu scikit-learn zbuduj model regresji liniowej. Wyucz model na danych
# treningowych i dokonaj oceny na danych testowych. Wynik wydrukuj do konsoli tak jak pokazano poniżej.

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
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

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

model = LinearRegression()
model.fit(data_train, target_train)
print(f'R^2 score: {model.score(data_test, target_test):.4f}')
