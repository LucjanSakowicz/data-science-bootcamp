# Ustaw opcje pakietu pandas pozwalające na wyświetlenie 15 kolumn obiektu DataFrame oraz wyświetlenie długości linii
# składającej się ze 150 znaków. Następnie wykorzystując funkcję load_boston() z pakietu scikit-learn załaduj dane do
# zmiennej raw_data. W oparciu o klucze 'data' oraz 'target' zmiennej raw_data przygotuj poniższy obiekt DataFrame: W
# odpowiedzi wydrukuj pięć pierwszych wierszy obiektu DataFrame.

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston

pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 150)
raw_data = load_boston()

df = pd.DataFrame(
    data=np.c_[raw_data.data, raw_data.target],
    columns=list(raw_data.feature_names) + ['target'],
)
print(df.head())