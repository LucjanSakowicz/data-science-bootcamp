# Do obiektu df przypisz nową kolumnę o nazwie 'number', która przyjmie liczbę elementów listy w kolumnie currency. W
# odpowiedzi wydrukuj obiekt DataFrame do konsoli.

import pandas as pd

data_dict = {
    'currency': [
        ['PLN', 'USD'],
        ['EUR', 'USD', 'PLN', 'CAD'],
        ['GBP'],
        ['JPY', 'CZK', 'HUF'],
        [],
    ]
}
df = pd.DataFrame(data=data_dict)
df['number'] = df['currency'].apply(len)
print(df)
