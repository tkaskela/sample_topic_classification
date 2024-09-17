import pandas as pd
import re
file = 'file.csv'
df = pd.read_csv(file)

print(df.columns)
print(len(df['tweet']))


def tokenize(doc):
    WORD = re.compile(r'\w+')
    tokens = WORD.findall(doc)
    return tokens


word_length = [len(tokenize(word)) for word in df['tweet'].str.lower()]
print(word_length)

df['tweet_length'] = word_length
df = df[['url', 'tweet_length']]
df.to_csv('tweet_length.csv', index=False)
