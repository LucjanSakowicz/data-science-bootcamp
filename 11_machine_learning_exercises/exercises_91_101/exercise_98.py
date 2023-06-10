# Przygotowano poniższe zbiory: data_train, target_train data_test, target_test Wykorzystano klasę LinearRegression z
# pakietu scikit-learn do zbudowania modelu regresji liniowej. Wyuczono model na danych treningowych. Dokonaj
# predykcji na podstawie modelu na danych testowych i wynik przypisz do zmiennej target_pred. Wydrukuj zmienną
# target_pred do konsoli.

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


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

regressor = LinearRegression()
regressor.fit(data_train, target_train)
target_pred = regressor.predict(data_test)
print(target_pred)