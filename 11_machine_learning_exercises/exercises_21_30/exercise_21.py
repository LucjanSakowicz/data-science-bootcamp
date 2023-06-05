# Podane są poniższe tablice: data_train, data_test target_train, target_test Zbuduj model regresji logistycznej (
# ustaw tylko argument max_iter=100, resztę pozostaw domyślnie) wykorzystując pakiet scikit-learn oraz dane IRIS.
# Model wytrenuj na danych treningowych i następnie dokonaj oceny modelu na zbiorze testowym. W odpowiedzi wydrukuj
# dokładność modelu na zbiorze testowym tak jak pokazano poniżej.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.3, random_state=20
)

clf = LogisticRegression(max_iter=100).fit(data_train, target_train)
print(f'Accuracy: {round(clf.score(data_test, target_test), 4)}')