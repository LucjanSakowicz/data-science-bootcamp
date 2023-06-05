# Podziel wartości kolumny hashtags względem znaku hash '#' używając pd.Series.str.split() z argumentem expand=True. Otrzymasz cztery kolumny.
# Przykładowo dla wiersza drugiego:
# '#hot#summer#holiday'
# odpowiednio:
# ['', 'hot', 'summer', 'holiday']
# Usuń pierwszą kolumnę z tak otrzymanego obiektu (pierwsza kolumna zawiera puste stringi). Następnie przypisz nazwy pozostałych kolumn odpowiednio:
# 'hashtag1'
# 'hashtag2'
# 'hashtag3'
# Postać końcową obiektu df wydrukuj do konsoli.

import pandas as pd


df = pd.DataFrame(
    data={
        'hashtags': [
            '#good#vibes',
            '#hot#summer#holiday',
            '#street#food',
            '#workout',
        ]
    }
)
df = df['hashtags'].str.split('#', expand=True)
df.drop(columns=df.columns[0], axis=1, inplace=True)
df.columns = ['hashtag1', 'hashtag2', 'hashtag3']
print(df)