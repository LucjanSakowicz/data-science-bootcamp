# Podane są poniższe tablice: data_train, data_test target_train, target_test Zbudowano model regresji logistycznej
# wykorzystując pakiet scikit-learn oraz dane IRIS. Dokonaj predykcji danych testowych na podstawie modelu i przypisz
# do zmiennej target_pred. W odpowiedzi wyświetl zmienną target_pred do konsoli.


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


data_raw = load_iris()
data = data_raw['data']
target = data_raw['target']
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.3, random_state=20
)

model = LogisticRegression(max_iter=1000)
model.fit(data_train, target_train)
target_pred = model.predict(data_test)
print(target_pred)