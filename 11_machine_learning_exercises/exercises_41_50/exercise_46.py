# W pliku predictions.csv znajdują się wyniki predykcji na podstawie pewnego modelu klasyfikacji wieloklasowej (3
# klasy). Kolumna y_true opisuje rzeczywiste wartości, zaś kolumna y_pred wartości przewidziane przez model.
# Wykorzystując funkcję confusion_matrix() z pakitu scikit-learn wyznacz macierz konfuzji i wydrukuj ją do konsoli.

import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_csv('predictions_45.csv')
print(confusion_matrix(df.y_true, df.y_pred))
