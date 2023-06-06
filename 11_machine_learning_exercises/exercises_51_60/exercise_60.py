# Wykorzystując klasę KMeans z pakietu scikit-learn dokonano podziału danych na trzy klastry. Dokonaj predykcji na
# podstawie zbudowanego modelu kmeans i przypisz numer klastra do każdej próbki w obiekcie df (nadaj nazwę kolumny
# 'y_kmeans'). W odpowiedzi wyświetl dziesięć pierwszych wierszy obiektu df.

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


np.random.seed(42)
df = pd.read_csv('clusters_60.csv')

kmeans = KMeans(n_clusters=3, max_iter=1000, random_state=42)
kmeans.fit(df)
df['y_kmeans'] = kmeans.predict(df)
print(df.head(10))
