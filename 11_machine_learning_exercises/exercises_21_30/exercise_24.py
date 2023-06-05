# Podane są poniższe tablice: data_train, data_test target_train, target_test Zbudowano model regresji logistycznej
# wykorzystując pakiet scikit-learn oraz dane IRIS. Dokonano predykcji danych testowych na podstawie modelu i
# przypisano do zmiennej target_pred. Wyświetl raport klasyfikacji modelu wykorzystując funkcję
# classification_report() z pakietu scikit-learn.

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.3, random_state=20
)

model = LogisticRegression(max_iter=1000)
model.fit(data_train, target_train)

target_pred = model.predict(data_test)
print(classification_report(target_test, target_pred))
