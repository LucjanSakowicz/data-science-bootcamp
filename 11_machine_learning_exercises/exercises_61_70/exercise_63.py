# Wykorzystując klasę AgglomerativeClustering z pakietu scikit-learn dokonaj podziału danych na dwa klastry. Dokonaj
# predykcji na podstawie zbudowanego modelu i przypisz numer klastra do każdej próbki w obiekcie df (nadaj nazwę
# kolumny cluster). Wyświetl dziesięć pierwszych wierszy obiektu df.

import pandas as pd
from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering()
df = pd.read_csv('clusters_63.csv')

df['cluster'] = model.fit_predict(df)
print(df.head(10))
