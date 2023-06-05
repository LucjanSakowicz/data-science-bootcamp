# Wczytaj plik data.csv do obiektu DataFrame. Następnie dokonaj ekstrakcji cech wielomianowych ze zmiennych var1 oraz
# var2 stopnia trzeciego. Otrzymane cechy w postaci tablicy numpy wyświetl do konsoli.

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


np.set_printoptions(suppress=True, precision=3, linewidth=150)
df = pd.read_csv(
    'data_38.csv'
)
print(PolynomialFeatures(degree=3).fit_transform(df))