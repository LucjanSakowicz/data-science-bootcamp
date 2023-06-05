# Załadowano zbiór IRIS wykorzystując pakiet scikit-learn do zmiennej data. Wyświetl nazwy zmiennych (klucz
# 'feature_names') oraz nazwy klas (klucz 'target_names') w zbiorze IRIS tak jak pokazano poniżej.

from sklearn.datasets import load_iris


data = load_iris()
print(data['feature_names'])
print(data['target_names'])