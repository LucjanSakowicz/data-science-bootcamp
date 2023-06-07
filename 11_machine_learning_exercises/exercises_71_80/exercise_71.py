# Wczytaj plik pca.csv do obiektu DataFrame df. Plik zawiera zmienne var_1, ..., var_10. Dokonaj analizy PCA z trzema
# komponentami wykorzystując pakiet scikit-learn oraz klasę PCA. W odpowiedzi wydrukuj procent wyjaśnionej wariancji
# przez te komponenty tak jak pokazano poniżej.


import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(42)
df = pd.read_csv('pca_71.csv')


scaler = StandardScaler()
X_std = scaler.fit_transform(df)

pca = PCA(n_components=3)
pca.fit(X_std)

results = pd.DataFrame(
    data={'explained_variance_ratio': pca.explained_variance_ratio_}
)
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
print(results)