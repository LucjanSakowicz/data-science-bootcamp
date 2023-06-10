# Wykorzystując klasę IsolationForest z pakietu scikit-learn dokonano analizy elementów odstających na podanym
# zbiorze. Dla przypomnienia 1 oznacza normalny element, -1 element odstający. Przypisano nową kolumnę do obiektu df
# o nazwie 'outlier_flag', która przechowuje informację czy dana próbka jest normalnym czy odstającym elementem:
# Zbadaj liczbę elementów odstających w zbiorze, tzn. zbadaj rozkład kolumny outlier_flag. Wynik wydrukuj do konsoli.

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


np.random.seed(42)
df = pd.read_csv('factory_82.csv')

outlier = IsolationForest(
    n_estimators=100, contamination=0.05, random_state=42
)
outlier.fit(df)
df['outlier_flag'] = outlier.predict(df)

print(df['outlier_flag'].value_counts())
