# Wczytaj plik pca.csv do obiektu DataFrame df. Plik zawiera trzy zmienne objaśniające var1, var2, var3 oraz zmienną
# docelową class. Następnie przypisz do zmiennej X kolumny: var1, var2, var3, zaś do zmiennej y kolumnę class.
# Wykorzystując klasę StandardScaler dokonaj standaryzacji zmiennych w obiekcie X. Wyświetl dziesięć pierwszych
# wierszy obiektu X.


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
df = pd.read_csv('pca_67.csv')

X = df.copy()
y = X.pop('class')

scaler = StandardScaler()
std = scaler.fit_transform(X)
print(std[:10])
