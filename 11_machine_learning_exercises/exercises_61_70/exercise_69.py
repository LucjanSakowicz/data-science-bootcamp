# Wczytano plik pca.csv do obiektu DataFrame df. Wykorzystując klasę StandardScaler dokonano standaryzacji zmiennych
# podanych w pliku i przypisano do zmiennej X_std. Zaimplementowano algorytm PCA wykorzystując tablicę X_std i
# przypisano do zmiennej X_pca. Zbuduj obiekt DataFrame o nazwie df_pca wykorzystując tablicę X_pca oraz zmienną y
# tak jak pokazano poniżej. W odpowiedzi wydrukuj dziesięć pierwszych wierszy tego obiektu do konsoli.

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


np.set_printoptions(
    precision=8, suppress=True, edgeitems=5, linewidth=200
)
np.random.seed(42)
df = pd.read_csv('pca_69.csv')

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

df_pca = pd.DataFrame(data=np.c_[X_pca, y], columns =['pca_1', 'pca_2', 'class'])
#TODO dlaczego minus?
df_pca['pca_2'] = -df_pca['pca_2']
print(df_pca.head(10))