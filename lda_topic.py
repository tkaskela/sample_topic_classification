from sklearn.feature_extraction.text import CountVectorizer
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from datetime import datetime
import nltk
import pandas as pd
import re
import math
import matplotlib.pyplot as plt
import matplotlib

nltk.download('stopwords')


def clean_tweets(df, tweet_col='tweet'):
    df_copy = df.copy()
    # lower the tweets
    df_copy['preprocessed_' + tweet_col] = df_copy[tweet_col].str.lower()
    # filter out stop words and URLs
    print(df_copy['preprocessed_' + tweet_col])
    en_stop_words = set(stopwords.words('english'))
    extended_stop_words = en_stop_words | \
                          {
                              '&amp;', 'rt',
                              'th', 'co', 're', 've', 'kim', 'daca', 'via', 're', 'fe0f'
                          }
    url_re = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    df_copy['preprocessed_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: ' '.join(
        [word for word in row.split() if (not word in extended_stop_words) and (not re.match(url_re, word))]))
    # tokenize the tweets
    tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
    df_copy['tokenized_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: tokenizer.tokenize(row))
    print(df_copy)
    return df_copy


def get_most_freq_words(str, n=None):
    vect = CountVectorizer().fit(str)
    bag_of_words = vect.transform(str)
    sum_words = bag_of_words.sum(axis=0)
    freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
    freq = sorted(freq, key=lambda x: x[1], reverse=True)
    return freq[:n]


file = 'file.csv'
df_nasdaq = pd.read_csv(file)
print(df_nasdaq['tweet'])
df_nasdaq = clean_tweets(df_nasdaq)
print(len(df_nasdaq))
print(df_nasdaq['tweet'])
print(get_most_freq_words([word for tweet in df_nasdaq['tokenized_tweet'] for word in tweet], 10))


# build a dictionary where for each tweet, each word has its own id.
tweets_dictionary = Dictionary(df_nasdaq['tokenized_tweet'])
tweets_dictionary.filter_extremes(no_below=20, no_above=0.5)

# build the corpus i.e. vectors with the number of occurence of each word per tweet
tweets_corpus = [tweets_dictionary.doc2bow(tweet) for tweet in df_nasdaq['tokenized_tweet']]

# compute coherence
tweets_coherence = []
topic_start = 5
topic_num = 250
for nb_topics in range(topic_start,topic_num):
    print('Round: ' + str(nb_topics))
    lda = LdaModel(tweets_corpus, num_topics=nb_topics, id2word=tweets_dictionary, passes=10)
    cohm = CoherenceModel(model=lda, corpus=tweets_corpus, dictionary=tweets_dictionary, coherence='u_mass')
    coh = cohm.get_coherence()
    tweets_coherence.append(coh)

# visualize coherence
plt.figure(figsize=(10,5))
plt.plot(range(topic_start,topic_num),tweets_coherence)
tweet_df = pd.DataFrame(tweets_coherence)
tweet_df.to_csv('tweets_coherence.csv')
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.show()

#k = max(tweets_coherence)
k = tweets_coherence.index(max(tweets_coherence)) + 5
tweets_lda = LdaModel(tweets_corpus, num_topics=k, id2word=tweets_dictionary, passes=10)
tweets_lda.save('NASDAQ_lda')
tweets_dictionary.save('dictionary')
topic_dict = {'count': [], 'post_id': [], 'index': [], 'combined_text': [], 'score': [], 'cleaned_text': []}

for i in range(len(tweets_corpus)):
    for index, score in sorted(tweets_lda[tweets_corpus[i]], key=lambda tup: -1 * tup[1]):
        topic_dict['count'].append(i)
        topic_dict['combined_text'].append(df_nasdaq['tweet'][i])
        topic_dict['post_id'].append(df_nasdaq['url'][i])
        topic_dict['index'].append(index)
        topic_dict['score'].append(score)
        topic_dict['cleaned_text'].append(df_nasdaq['tweet'][i])
        # print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

df = pd.DataFrame(topic_dict)
df.to_csv('four_full_topic.csv', encoding='utf-8', index=False)
unique_cnt = df['count'].unique()
df_c = {'post_id': [], 'index': [], 'combined_text': [], 'score': [], 'cleaned_text': [], 'url': []}
print(len(unique_cnt))
for a in range(len(unique_cnt)):
    df_b = df.loc[(df['count'] == a)]
    df_c['post_id'].append(df_b.loc[(df_b['score'] == df_b['score'].max())]['post_id'].astype('str').iloc[0])
    df_c['index'].append(df_b.loc[(df_b['score'] == df_b['score'].max())]['index'].to_frame().T.iloc[0].values[0])
    df_c['combined_text'].append(df_b.loc[(df_b['score'] == df_b['score'].max())]['combined_text'].to_frame().T.iloc[0].values[0])
    df_c['score'].append(df_b.loc[(df_b['score'] == df_b['score'].max())]['score'].to_frame().T.iloc[0].values[0])
    df_c['cleaned_text'].append(df_b.loc[(df_b['score'] == df_b['score'].max())]['cleaned_text'].to_frame().T.iloc[0].values[0])
print(len(df_c['post_id']))
print(len(df_c['index']))
print(len(df_c['combined_text']))
print(len(df_c['score']))
print(len(df_c['cleaned_text']))
df_d = pd.DataFrame(df_c)
df_d.to_csv('four_unique_all.csv', index=False, encoding='utf-8')


def plot_top_words(lda=tweets_lda, nb_topics=k, nb_words=10):
    top_words = [[word for word, _ in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]
    top_betas = [[beta for _, beta in lda.show_topic(topic_id, topn=50)] for topic_id in range(lda.num_topics)]

    gs = matplotlib.gridspec.GridSpec(round(math.sqrt(k)) + 1, round(math.sqrt(k)) + 1)
    gs.update(wspace=0.5, hspace=0.5)
    plt.figure(figsize=(20, 15))
    for i in range(nb_topics):
        ax = plt.subplot(gs[i])
        plt.barh(range(nb_words), top_betas[i][:nb_words], align='center', color='blue', ecolor='black')
        ax.invert_yaxis()
        ax.set_yticks(range(nb_words))
        ax.set_yticklabels(top_words[i][:nb_words])
        plt.title("Topic " + str(i))
        plt.show()


plot_top_words()
