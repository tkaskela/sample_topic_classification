import tweetnlp
import pandas as pd
from time import time

model = tweetnlp.load_model('sentiment')


def tweet_nlp_sentiment(text):
    print(text)
    sent_output = model.sentiment(text, return_probability=True)
    return sent_output['label'], sent_output['probability']['negative'], sent_output['probability']['positive'], sent_output['probability']['neutral']


start_time = time()
file = 'file.csv'
df = pd.read_csv(file)
#df = df.sample(frac=0.005).reset_index(drop=True)
df['test'] = [tweet_nlp_sentiment(row) for row in df['tweet'].values]
df[['nlp_sent_label', 'nlp_sent_neg', 'nlp_sent_neu', 'nlp_sent_pos']] = df['test'].tolist()
df.drop(['test', 'tweet'], axis=1, inplace=True)
df.to_csv('file_sentiment_tweetnlp.csv', index=False)
end_time = time()
print('Time took: ' + str(end_time - start_time))
