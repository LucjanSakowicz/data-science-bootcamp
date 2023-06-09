# Wykorzystując klasę LocalOutlierFactor z pakietu scikit-learn dokonaj analizy elementów odstających w podanym
# zbiorze. Ustaw argument: n_neighbors=20 Dla przypomnienia 1 oznacza normalny element, -1 element odstający.
# Przypisz nową kolumnę do obiektu df o nazwie 'lof', która będzie przechowywać informację czy dana próbka jest
# elementem normalnym czy odstającym. Wydrukuj dziesięć pierwszych wierszy obiektu df do konsoli.

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.read_csv('blobs_79.csv')
model = LocalOutlierFactor(n_neighbors=20)
df['lof'] = model.fit_predict(df)
LOF_scores = model.negative_outlier_factor_
radius = (LOF_scores.max() - LOF_scores) / (LOF_scores.max() - LOF_scores.min())

plt.figure(figsize=(12, 7))
plt.scatter(df['x2'], df['x1'], c=df['lof'], cmap='tab10', label='data')
plt.scatter(df['x2'], df['x1'], s=2000 * radius, edgecolors='r', facecolors='none', label='outlier scores')
plt.title('Local Outlier Factor')
legend = plt.legend()
legend.legendHandles[1]._sizes = [40]
plt.show()
print(df.head(10))
