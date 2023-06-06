# Wczytaj plik data.csv do obiektu DataFrame (plik zawiera dwie zmienne x1 oraz x2). Następnie zaimplementuj algorytm
# K-średnich pozwalających rozdzielić podane dane na dwa klastry. Wyznacz centroid każdego klastra i wydrukuj jego
# współrzędne do konsoli. Zaokrąglij wynik do trzech miejsc po przecinku każdej ze współrzędnych. Pomocnicze kroki:
# Wyznaczamy przedziały wartości dla zmiennych x1 oraz x2.
# Losowo wybieramy współrzędne centroidów z wyznaczonych przedziałów.
# Przyporządkowujemy punkty do najbliższego centroidu.
# Obliczamy nowe współrzędne centroidów.
# Wracamy do kroku 3 i powtarzamy do osiągnięcia zbieżności (wystarczy 10 iteracji).
# Do obliczenia odległości punktów wykorzystaj normę euklidesową - funkcja np.linalg.norm().


import numpy as np
from numpy.linalg import norm
import pandas as pd
import random

np.random.seed(42)
df = pd.read_csv('data_58.csv')

x1_min = df.x1.min()
x1_max = df.x1.max()

x2_min = df.x2.min()
x2_max = df.x2.max()

centroid_1 = np.array(
    [random.uniform(x1_min, x1_max), random.uniform(x2_min, x2_max)]
)
centroid_2 = np.array(
    [random.uniform(x1_min, x1_max), random.uniform(x2_min, x2_max)]
)

data = df.values

for i in range(10):
    clusters = []
    for point in data:
        centroid_1_dist = norm(centroid_1 - point)
        centroid_2_dist = norm(centroid_2 - point)
        cluster = 1
        if centroid_1_dist > centroid_2_dist:
            cluster = 2
        clusters.append(cluster)

    df['cluster'] = clusters

    centroid_1 = [
        round(df[df.cluster == 1].x1.mean(), 3),
        round(df[df.cluster == 1].x2.mean(), 3),
    ]
    centroid_2 = [
        round(df[df.cluster == 2].x1.mean(), 3),
        round(df[df.cluster == 2].x2.mean(), 3),
    ]

print(centroid_1)
print(centroid_2)
