# Wykorzystując klasę IsolationForest z pakietu scikit-learn dokonaj analizy elementów odstających na podanym
# zbiorze. Przekaż argumenty: n_estimators=100 contamination=0.05 random_state=42 Dla przypomnienia 1 oznacza
# normalny element, -1 element odstający. Przypisz nową kolumnę do obiektu df o nazwie 'outlier_flag', która będzie
# przechowywać informację czy dana próbka jest elementem normalnym czy odstającym. Wydrukuj dziesięć pierwszych
# wierszy obiektu df do konsoli.


import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

np.random.seed(42)
df = pd.read_csv('factory_81.csv')

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

df['outlier_flag'] = model.fit_predict(df)

print(df.head(10))
