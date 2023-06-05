# Przypisz do obiektu df nową kolumnę o nazwie 'PLN_flag', która przyjmie wartość 1, gdy waluta 'PLN' będzie w liście
# w kolumnie currency i przeciwnie 0. W odpowiedzi wydrukuj obiekt DataFrame do konsoli.


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
df['PLN_flag'] = df['currency'].apply(
    lambda item: 1 if 'PLN' in item else 0
)
print(df)
