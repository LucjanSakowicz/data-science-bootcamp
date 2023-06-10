# Wykorzystując funkcję load_digits() z pakietu scikit-learn załaduj dane dotyczące obrazów o rozdzielczości 8x8
# pikseli do zmiennych: data - obrazy zapisane w postaci tablic numpy kształtu (64,) target - etykiety,
# cyfry widoczne na obrazach Zapoznaj się dokładnie z podanym zbiorem. Spróbuj wyświetlić kilka przykładowych
# obrazów. W celu wyświetlenia obrazu można użyć pakietu matplotlib następująco: Zmieniając wartość zmiennej idx
# wyświetl kilka obrazów. W odpowiedzi wydrukuj etykietę dla obrazu z indeksem 250.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


np.random.seed(42)
data, target = load_digits(return_X_y=True)

idx = 250
plt.imshow(data[idx].reshape(8, 8), cmap='gray_r')
plt.title(f'Label: {target[idx]}')
plt.show()
print(target[idx])