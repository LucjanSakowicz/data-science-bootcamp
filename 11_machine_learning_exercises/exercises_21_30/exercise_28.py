# Poniżej załadowano zbiór Breast Cancer Data do zmiennej raw_data. Przypisz do zmiennej data tablicę numpy z danymi
# ze zmiennej raw_data (zawartość klucza 'data') oraz do zmiennej target tablicę numpy ze zmienną docelową (zawartość
# klucza 'target'). W odpowiedzi wyświetl trzy pierwsze elementy tablicy data do konsoli.

import numpy as np
from sklearn.datasets import load_breast_cancer


np.set_printoptions(precision=2, suppress=True, linewidth=100)
raw_data = load_breast_cancer()

data = raw_data['data']
target = raw_data['target']
print(data[:3])