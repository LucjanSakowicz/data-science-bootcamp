# Z poniższego słownika data utwórz obiekt DataFrame i przypisz do zmiennej df. Następnie zapoznaj się z obiektem df
# i sprawdź liczbę braków danych dla wszystkich kolumn. Podaj procent braków, wynik zaokrąglij do drugiego miejsca po
# przecinku i wydrukuj do konsoli tak jak pokazano poniżej.

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

df=pd.DataFrame(data)
print(round(df.isnull().sum() / len(df), 2))