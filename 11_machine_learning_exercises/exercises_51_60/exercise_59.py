import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


np.random.seed(42)
df = pd.read_csv('clusters_59.csv')
model = KMeans(max_iter=1000, random_state=42, n_clusters=3)
model.fit_transform(df)
print(model.cluster_centers_)