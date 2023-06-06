# Wczytano plik pca.csv do obiektu DataFrame df. Wykorzystując klasę StandardScaler dokonano standaryzacji zmiennych
# podanych w pliku i przypisano do zmiennej X_std. Zaimplementuj algorytm PCA wykorzystując tablicę X_std. Wynik
# ogranicz do dwóch głównych komponentów PCA i przypisz do zmiennej X_pca. Wydrukuj dziesięć pierwszych wierszy
# obiektu X_pca. Pomocnicze kroki: Standaryzacja danych. Wyznaczenie macierzy kowariancji. Wyznaczenie wektorów i
# odpowiadających im wartości własnych macierzy kowariancji. Posortowanie wektorów własnych względem malejących
# wartości własnych. Ustalenie liczby komponentów (w tym przypadku 2). Wyznaczenie macierzy W z wybranych wektorów (
# kolumny jako wektory własne). Przemnożenie X_std przez macierz W.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

np.set_printoptions(
    precision=8, suppress=True, edgeitems=5, linewidth=200
)
np.random.seed(42)
df = pd.read_csv('pca_68.csv')

X = df.copy()
y = X.pop('class')

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

eig_vals, eig_vecs = np.linalg.eig(np.cov(X_std, rowvar=False))
eig_pairs = [
    (np.abs(eig_vals[i]), eig_vecs[:, i])
    for i in range(len(eig_vals))
]
eig_pairs.sort(reverse=True)

W = np.hstack(
    (eig_pairs[0][1].reshape(3, 1), eig_pairs[1][1].reshape(3, 1))
)
X_pca = X_std.dot(W)
print(X_pca[:10])