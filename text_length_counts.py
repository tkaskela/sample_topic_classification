import pandas as pd
import nltk
import re
from statistics import mean
import statistics

def word_counts(text):
    text_length = len(re.findall(r'\w+', text))
    return text_length


def char_count(text):
    six_count = 0
    text_split = re.findall(r'\w+', text)
    for word in text_split:
        if len(word) >= 6:
            six_count += 1
    return six_count


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

file = 'sentiment_df.csv'
df = pd.read_csv(file)

df['tweet'] = df['tweet'].replace(r'[^\w\s.,\/#!$%\^&\*;:{}=\-_`~()]', '', regex=True)
df['total_words'] = df['tweet'].str.split().str.len()
words_per_sentence = df['tweet'].astype(str).apply(lambda x: nltk.sent_tokenize(x))
words_avg = []
for sentences in words_per_sentence:
    avg = []
    for sentence in sentences:
        avg.append(word_counts(sentence))
    try:
        words_avg.append(mean(avg))
    except statistics.StatisticsError:
        words_avg.append(0)
df['words_per_sentence'] = words_avg
df['long_word_count'] = df['tweet'].astype(str).apply(lambda a: char_count(a))
df['long_word_perc'] = df['long_word_count'] / df['total_words']
df.drop(['tweet'], inplace=True, axis=1)
df.to_csv('file_words_per_sentence.csv', index=False)
