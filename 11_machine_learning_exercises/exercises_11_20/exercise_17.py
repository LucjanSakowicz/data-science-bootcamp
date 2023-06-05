# Załaduj zbiór danych IRIS do zmiennej data wykorzystując pakiet scikit-learn oraz funkcję load_iris(). Następnie
# wyświetl wszystkie klucze zmiennej data do konsoli.

from sklearn.datasets import load_iris
data = load_iris()
print(data.keys())