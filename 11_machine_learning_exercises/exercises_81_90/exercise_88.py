# Podane są dwa pliki: data_train.csv target_train.csv Plik data_train.csv zawiera maile dotyczące dwóch kategorii:
# grafiki komputerowej (comp.graphics) oraz przestrzeni kosmicznej (sci.space). Plik target_train.csv zawiera
# odpowiednio etykiety (0 - comp.graphics, 1 - sci.space). Wczytano zawartość plików jako obiekty DataFrame
# odpowiednio o nazwach: data_train target_train Pięć pierwszych wierszy obiektu data_train: text 0  From:
# ab@nova.cc.purdue.edu (Allen B)\nSubject... 1  From: stephens@geod.emr.ca (Dave Stephenson)\n... 2  From:
# dotzlaw@ccu.umanitoba.ca (Helmut Dotzlaw... 3  From: flb@flb.optiplan.fi ("F.Baube[tm]")\nSub... 4  From:
# cchung@sneezy.phy.duke.edu (Charles Chun... Przekształć obiekt data_train do postaci listy. Przypisz zmiany na
# trwałe do zmiennej data_train. W odpowiedzi wydrukuj długość listy data_train do konsoli.

import pandas as pd

data_train = pd.read_csv('data_train_88.csv')
target_train = pd.read_csv('target_train_88.csv')
data_train = data_train['text'].tolist()
print(data_train)