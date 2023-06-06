# Wczytaj pliki X_train.csv oraz X_test.csv odpowiednio do obiektów DataFrame o nazwach X_train oraz X_test.
# Następnie wykorzystując klasę StandardScaler z pakietu scikit-learn dokonaj standaryzacji danych. Dane dopasuj na
# zbiorze treningowym X_train. W odpowiedzi wydrukuj pięć pierwszych wierszy tak przetworzonych obiektów X_train oraz
# X_test.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


np.set_printoptions(precision=4, suppress=True)
X_train = pd.read_csv('X_train43.csv')
X_test = pd.read_csv('X_test43.csv')
scaler = StandardScaler()
scaler.fit(X_train)
print(scaler.transform(X_train)[:5])
print(scaler.transform(X_test)[:5])