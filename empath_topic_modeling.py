from empath import Empath
from time import time
import pandas as pd

lexicon = Empath()

start_time = time()
file = 'file.csv'
df = pd.read_csv(file)
df['tweet_topic'] = [lexicon.analyze(row, normalize=True) for row in df['tweet'].values]
print(df['tweet_topic'][0])
df = pd.concat([df.drop(['tweet_topic', 'tweet'], axis=1), df['tweet_topic'].apply(pd.Series)], axis=1)

df.to_csv('file_empath.csv', index=False)
end_time = time()
print('Time took: ' + str(end_time - start_time))
