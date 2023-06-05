# Podane są poniższe tablice:
# data
# target
# Połącz te dwie tablice w jedną o nazwie all_data i wydrukuj trzy pierwsze wiersze tej tablicy do konsoli.

import numpy as np
from sklearn.datasets import load_breast_cancer

np.set_printoptions(precision=2, suppress=True, linewidth=100)
raw_data = load_breast_cancer()

data = raw_data['data']
target = raw_data['target']

all_data = np.c_[data, target]
print(all_data[:3])
