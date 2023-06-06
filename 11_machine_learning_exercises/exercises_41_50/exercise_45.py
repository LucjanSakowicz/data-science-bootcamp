# W pliku predictions.csv znajdują się wyniki predykcji na podstawie pewnego modelu klasyfikacji wieloklasowej (3
# klasy). Kolumna y_true opisuje rzeczywiste wartości, zaś kolumna y_pred wartości przewidziane przez model.
# Wykorzystując funkcję accuracy_score() z pakietu scikit-learn policz dokładność tego modelu. Wynik z dokładnością
# do czterech miejsc po przecinku wyświetl do konsoli.

import pandas as pd
from sklearn.metrics import accuracy_score

df = pd.read_csv('predictions_45.csv')
print(f'Accuracy: {accuracy_score(df.y_true, df.y_pred):.4f}')
