# Plik predictions.csv zawiera predykcje pewnego modelu regresji: zmienna y_true opisuje rzeczywiste, zaobserwowane
# wartości zmienna y_pred opisuje wartości przewidziane przez model Wczytaj ten plik do obiektu DataFrame. Następnie
# zaimplementuj funkcję o nazwie mean_absolute_error() obliczającą średni błąd bezwzględny predykcji. Wykorzystując
# zaimplementowaną funkcję policz wartość MAE dla tego modelu. Wynik wydrukuj do konsoli tak jak pokazano poniżej.


import pandas as pd


def mean_absolute_error(y_true, y_pred):
    return abs(y_true - y_pred).sum() / len(y_true)


df = pd.DataFrame(pd.read_csv('predictions.csv'))
mae = mean_absolute_error(df['y_true'], df['y_pred'])
print(f'MAE = {mae:.4f}')
