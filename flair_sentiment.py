import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
from sklearn.metrics import accuracy_score
from time import time
sia = TextClassifier.load('en-sentiment')


def flair_sentiment(text):
    s = Sentence(str(text))
    sia.predict(s)
    print(s)
    total_sentiment = s.labels[0]
    assert total_sentiment.value in ['POSITIVE', 'NEGATIVE']
    sign = 1 if total_sentiment.value == 'POSITIVE' else -1
    score = total_sentiment.score
    total_score = sign * score
    if "POSITIVE" in str(total_sentiment):
        check = "positive"
    elif "NEGATIVE" in str(total_sentiment):
        check = "negative"
    else:
        check = "neutral"
    return total_score, check


start_time = time()
file = 'file.csv'
df = pd.read_csv(file)
df['tweet_flair_sentiment'] = [flair_sentiment(row)[0] for row in df['tweet'].values]
df['tweet_flair_posneg'] = [flair_sentiment(row)[1] for row in df['tweet'].values]
df.to_csv('nasdaq_sentiment.csv', index=False)
end_time = time()
print('Time took: ' + str(end_time - start_time))
