# Wykorzystując klasę KMeans (ustaw parametr random_state=42) z pakietu scikit-learn wyznacz listę wartości WCSS (
# Within-Cluster Sum-of-Squared) dla liczby klastrów od 2 do 9 włącznie. Wartości WCSS zaokrąglij do drugiego miejsca
# po przecinku. Listę wydrukuj do konsoli.

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

np.random.seed(42)
df = pd.read_csv('clusters_61.csv')

wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    wcss.append(round(kmeans.inertia_, 2))
print(wcss)
