# Przygotowano poniższe zbiory: data_train, target_train data_test, target_test Wykorzystano klasę
# GradientBoostingRegressor z pakietu scikit-learn do zbudowania model regresji. Wyuczono model na danych
# treningowych. Zapisz model (zmienna regressor) do pliku o nazwie 'model.pkl' wykorzystując moduł pickle. Następnie
# wczytaj plik model.pkl do zmiennej regressor_loaded. W odpowiedzi wydrukuj do konsoli informacje o obiekcie
# regressor_loaded wykonując poniższy kod:

import pickle

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


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

regressor = GradientBoostingRegressor()
regressor.fit(data_train, target_train)

pickle.dump(regressor, open('model.pkl', 'wb'))
regressor_loaded = pickle.load(open('model.pkl', 'rb'))
print(regressor_loaded)