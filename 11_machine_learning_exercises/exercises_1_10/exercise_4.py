# Wykorzystując pakiet do uczenia maszynowego scikit-learn oraz klasę SimpleImputer uzupełnij braki danych dla kolumny
# price stałą wartością 99.0. Zmiany przypisz na stałe do obiektu df i wydrukuj ten obiekt do konsoli.

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data=data)
imp_mean = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=99.0)
df[['price']] = imp_mean.fit_transform(df[['price']])
print(df)

