# Przygotowano poniższe zbiory: data_train, target_train data_test, target_test Wykorzystano klasę LinearRegression z
# pakietu scikit-learn do zbudowania modelu regresji liniowej. Wyuczono model na danych treningowych. Dokonano
# predykcji na podstawie modelu na danych testowych i wynik przypisano do zmiennej target_pred. Zbuduj nowy obiekt
# DataFrame o nazwie predictions, który będzie przechowywał cztery kolumny: target_test target_pred error (różnica
# pomiędzy target_pred oraz target_test) abs_error (wartość bezwzględna z kolumny error) W odpowiedzi wydrukuj
# dziesięć pierwszych wierszy obiektu predictions do konsoli.

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

regressor = LinearRegression()
regressor.fit(data_train, target_train)

target_pred = regressor.predict(data_test)

predictions = pd.DataFrame(
    data=np.c_[target_test, target_pred, target_pred - target_test, abs(target_pred - target_test)],
    columns=['target_test', 'target_pred', 'error', 'abs_error']
)
print(predictions.head(10))
