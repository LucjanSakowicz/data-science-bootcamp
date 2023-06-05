# Podane są poniższe tablice: data target Wykorzystując pakiet scikit-learn oraz funkcję train_test_split() podziel
# dane na zbiór treningowy (data_train, target_train) i testowy (data_test, target_test). Ustal rozmiar zbioru
# testowego na 30% próbek.
# W odpowiedzi wyświetl rozmiar poniższych tablic (tak jak pokazano poniżej):
# data_train
# target_train
# data_test
# target_test


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']

data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.3
)

print(f'data_train shape: {data_train.shape}')
print(f'target_train shape: {target_train.shape}')
print(f'data_test shape: {data_test.shape}')
print(f'target_test shape: {target_test.shape}')
