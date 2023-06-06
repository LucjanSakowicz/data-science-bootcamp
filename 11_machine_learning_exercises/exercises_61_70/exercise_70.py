# Wczytano plik pca.csv do obiektu DataFrame df. Wykorzystując klasę StandardScaler dokonano standaryzacji zmiennych
# podanych w pliku i przypisano do zmiennej X_std. Wykorzystując klasę PCA z pakietu scikit-learn dokonaj analizy PCA
# z dwoma komponentami na obiekcie X_std i przypisz do zmiennej df_pca. Wydrukuj dziesięć pierwszych wierszy tego
# obiektu (dodaj także kolumnę class) tak jak pokazano poniżej.

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.set_printoptions(
    precision=8, suppress=True, edgeitems=5, linewidth=200
)
np.random.seed(42)
df = pd.read_csv('pca_70.csv')

X = df.copy()
y = X.pop('class')

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=2)
df_pca = pca.fit_transform(X_std)
df_pca = pd.DataFrame(data=np.c_[df_pca, y], columns =['pca_1', 'pca_2', 'class'])
print(df_pca.head(10))
