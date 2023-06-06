# Wczytaj plik data.csv do obiektu DataFrame. Następnie wykorzystując klasę StandardScaler z pakietu scikit-learn
# dokonaj standaryzacji wszystkich kolumn. Tak otrzymany obiekt DataFrame wyświetl do konsoli.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


np.set_printoptions(precision=4, suppress=True)
df = pd.read_csv('data_42.csv')
scaler = StandardScaler()
print(scaler.fit_transform(df))