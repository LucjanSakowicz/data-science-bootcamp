# Wczytaj plik data.csv do obiektu DataFrame. Następnie dokonaj ekstrakcji cech wielomianowych ze zmiennej var1
# stopnia drugiego. Otrzymane cechy w postaci tablicy numpy wyświetl do konsoli.

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


np.set_printoptions(suppress=True, precision=3)
df = pd.read_csv(
    'data_37.csv'
)
print(PolynomialFeatures(degree=2).fit_transform(df[['var1']]))