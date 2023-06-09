# Każdy wiersz zawiera produkty zakupione przez jednego klienta. Podzielono każdy wiersz kolumny products względem
# znaku spacji i rozszerzono do obiektu DataFrame (zmienna expanded) tak jak pokazano poniżej: Do zmiennej products
# przypisz unikalne nazwy wszystkich produktów występujących w bazie transakcji posortowanych alfabetycznie. Wydrukuj
# zmienną products do konsoli.

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
expanded = transactions['products'].str.split(expand=True)
products = sorted([i for i in set(expanded.to_numpy().ravel()) if i is not None])
print(products)
