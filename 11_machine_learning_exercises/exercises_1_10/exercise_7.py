# Wydobądź z obiektu df kolumny typu object. Następnie uzupełnij wszystkie braki dla tych kolumn wartością 'empty'.
# Przypisz wynik do zmiennej df_object i wydrukuj tą zmienną do konsoli.

import numpy as np
import pandas as pd


data = {
    'size': ['XL', 'L', 'M', np.nan, 'M', 'M'],
    'color': ['red', 'green', 'blue', 'green', 'red', 'green'],
    'gender': ['female', 'male', np.nan, 'female', 'female', 'male'],
    'price': [199.0, 89.0, np.nan, 129.0, 79.0, 89.0],
    'weight': [500, 450, 300, np.nan, 410, np.nan],
    'bought': ['yes', 'no', 'yes', 'no', 'yes', 'no']
}

df = pd.DataFrame(data=data)
df_object = df.select_dtypes(include=['object']).fillna('empty')
print(df_object)