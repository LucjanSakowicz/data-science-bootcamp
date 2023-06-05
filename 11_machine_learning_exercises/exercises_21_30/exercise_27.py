# Załaduj dane Breast Cancer Data wykorzystując funkcję load_breast_cancer() z pakietu scikit-learn do zmiennej
# raw_data. Następnie wydrukuj informacje o tym zbiorze do konsoli (zawartość klucza 'DESCR').

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
print(data['DESCR'])