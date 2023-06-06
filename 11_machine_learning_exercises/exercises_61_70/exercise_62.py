# Wykorzystując klasę KMeans z pakietu scikit-learn wyznaczono listę wartości WCSS (Within-Cluster Sum-of-Squared)
# dla liczby klastrów od 2 do 9 włącznie. Wykorzystując metodę łokcia (elbow method) dokonaj wyboru odpowiedniej
# liczby klastrów (najlepiej stwórz pomocniczy wykres WCSS). Wynik wydrukuj do konsoli.

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

np.random.seed(42)
df = pd.read_csv('clusters_62.csv')

wcss = []
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(df)
    wcss.append(round(kmeans.inertia_, 2))

wcss = pd.DataFrame(data=np.c_[range(2, 10), wcss], columns=['NumberOfClusters', 'WCSS'])
px.line(wcss, x='NumberOfClusters', y='WCSS', template='plotly_dark', title='WCSS',
        width=950, color_discrete_sequence=['#03fcb5']).show()
print(3)
