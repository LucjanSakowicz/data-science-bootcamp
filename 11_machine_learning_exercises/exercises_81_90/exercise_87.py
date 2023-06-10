# Podane są dwa pliki: data_train.csv target_train.csv Plik data_train.csv zawiera maile dotyczące dwóch kategorii:
# grafiki komputerowej (comp.graphics) oraz przestrzeni kosmicznej (sci.space). Plik target_train.csv zawiera
# odpowiednio etykiety (0 - comp.graphics, 1 - sci.space). Wczytaj zawartość plików jako obiekty DataFrame
# odpowiednio o nazwach: data_train target_train Zapoznaj się z danymi. W odpowiedzi wydrukuj zawartość drugiego
# elementu obiektu data_train.

import numpy as np
import pandas as pd

data_train = pd.read_csv('data_train_87.csv')
target_train = pd.read_csv('target_train_87.csv')
print(data_train['text'][1])
