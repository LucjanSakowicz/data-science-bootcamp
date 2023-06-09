# Każdy wiersz zawiera produkty zakupione przez jednego klienta. Podziel każdy wiersz kolumny products względem znaku
# spacji i rozszerz do obiektu DataFrame. Obiekt docelowo będzie posiadał 4 kolumny (maksymalna liczba produktów w
# jednej transakcji). W brakujące miejsca wpisz wartość None, tak jak pokazano poniżej i przypisz do zmiennej expanded.

import numpy as np
import pandas as pd


data = {
    'products': [
        'bread eggs',
        'bread eggs milk',
        'milk cheese',
        'bread butter cheese',
        'eggs milk',
        'bread milk butter cheese',
    ]
}

transactions = pd.DataFrame(data=data, index=range(1, 7))
expanded = transactions['products'].str.split(' ', expand=True)
print(expanded)