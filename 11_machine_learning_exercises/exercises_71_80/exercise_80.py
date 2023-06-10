import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)
df = pd.read_csv('blobs_80.csv')
data = df.values

lof = LocalOutlierFactor(n_neighbors=30)
y_pred = lof.fit_predict(data)

df['lof'] = y_pred
print(df['lof'].value_counts())
