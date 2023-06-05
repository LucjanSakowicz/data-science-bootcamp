#Braki danych dla kolumny weight zastąpiono średnią wartością tej kolumny wykorzystując klasę SimpleImputer z
#pakietu scikit-learn. Wyświetl wartość średnią wstawioną w miejsce braków dla tej kolumny
# (wykorzystaj instancję klasy SimpleImputer).

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
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
df[['weight']] = imp_mean.fit_transform(df[['weight']])
print(imp_mean.statistics_[0])

