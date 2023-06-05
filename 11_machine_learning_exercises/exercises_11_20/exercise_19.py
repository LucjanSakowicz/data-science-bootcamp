# Poniżej załadowano zbiór IRIS wykorzystując pakiet scikit-learn do zmiennej data_raw. Do zmiennej data przypisz
# dane zbioru IRIS (klucz 'data'). Do zmiennej target przypisz wartości zmiennej docelowej (klucz 'target') ze zbioru
# IRIS. W odpowiedzi wydrukuj kształt zmiennych: data oraz target do konsoli.

from sklearn.datasets import load_iris


data_raw = load_iris()

data = data_raw['data']
target = data_raw['target']
print(data.shape)
print(target.shape)