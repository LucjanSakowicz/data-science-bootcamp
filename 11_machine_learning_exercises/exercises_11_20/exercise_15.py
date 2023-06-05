# Utwórz nową kolumnę o nazwie 'missing' w obiekcie df i przypisz do niej liczbę brakujących hashtagów dla każdego
# wiersza. Przykładowo, wiersz pierwszy -> 1, wiersz drugi -> 0, wiersz trzeci -> 1, itd. W odpowiedzi wydrukuj
# obiekt df do konsoli.

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
df = df.drop(columns=[0])
df.columns = ['hashtag1', 'hashtag2', 'hashtag3']
df['missing'] = df.isnull().sum(axis=1)
print(df)