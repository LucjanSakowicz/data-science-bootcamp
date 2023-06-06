# Wykorzystując klasę DBSCAN z pakietu scikit-learn dokonaj podziału danych na klastry. Ustaw odpowiednio argumenty:
# eps=0.6 min_samples=7 Dokonaj predykcji na podstawie tak zbudowanego modelu i przypisz numer klastra do każdej
# próbki w obiekcie df (nadaj nazwę kolumny cluster). Wyświetl dziesięć pierwszych wierszy obiektu df.


import pandas as pd
from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.6,
    min_samples=7
)
df = pd.read_csv('clusters_65.csv')

df['cluster'] = model.fit_predict(df)
print(df.head(10))
