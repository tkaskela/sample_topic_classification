from bertopic import BERTopic
import pandas as pd
from umap.umap_ import UMAP
from nltk.tokenize.toktok import ToktokTokenizer
import nltk
import os
from hdbscan import HDBSCAN
from scipy import spatial
from scipy.stats import entropy

tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.extend(['say', 'use', 'go', 'also', 'get', 'place', 'order', 'amp'])


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

directory = 'bertopic_files'

for file_name in os.listdir(directory):
    if file_name.endswith('.csv'):
        umap_model = UMAP(n_components=50,
                                    densmap=True,
                                    metric='cosine',
                                    n_neighbors=15)

        hdbscan_model = HDBSCAN(min_cluster_size=3, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)
        #Need to strip all company references for each company
        model = BERTopic(embedding_model='all-mpnet-base-v2', umap_model=umap_model, hdbscan_model=hdbscan_model, calculate_probabilities=True)
        file = os.path.join(directory, file_name)
        df = pd.read_csv(file)
        df['tweet'] = [remove_stopwords(row) for row in df['tweet']]
        df['tweet'] = df.apply(lambda x: x['tweet'].replace(x['username'], ''), axis=1)
        df = df.dropna(subset=['tweet'])
        topics, probabilities = model.fit_transform(df['tweet'])
        #topics = model.reduce_outliers(df['tweet'], topics)
        save_file = file_name.split('.')[0].split('_')[0] + '_entropy_bertopic_test.csv'
        print(model.get_topic_info())
        topic_df = model.get_topic_info()
        topic_df.to_csv('topic_' + save_file, index=False)
        df['topics'] = topics
        print(len(df))
        #df['topics_check'] = model.topics_
        df_prob = pd.DataFrame(probabilities)
        df_prob = df_prob.add_prefix('topic_')
        print(len(df_prob))
        df_prob_check = df_prob.expanding().mean()
        df_prob_check = df_prob_check.add_prefix('mean_topic_')
        print(df_prob.expanding().mean())
        dist_list = [0]
        for i in range(1, len(df_prob)):
            distance = spatial.distance.cosine(df_prob.iloc[i], df_prob_check.iloc[i-1])
            dist_list.append(distance)
        print(entropy(df_prob.iloc[0]))
        df_prob = df_prob.astype(float)
        entropy_list = []
        df['total_cosine_distance'] = dist_list
        for row in range(0, len(df_prob)):
            entropy_list.append(entropy(df_prob.iloc[row]))
        #df['topic_complexity'] = [entropy(row) for row in df_prob]
        df['date'] = pd.to_datetime(df['date'])
        df_date = df['date']
        df_prob = df_prob.set_index(df_date)
        df_prob = df_prob.sort_index()
        df_rolling = df_prob.rolling('7D').mean()
        neigh_dist_list = [0]
        for i in range(1, len(df_prob)):
            neigh_distance = spatial.distance.cosine(df_prob.iloc[i], df_rolling.iloc[i-1])
            neigh_dist_list.append(neigh_distance)
        df['neighborhood_one_week_cosine_distance'] = neigh_dist_list
        df_rolling = df_prob.rolling('14D').mean()
        neigh_dist_list = [0]
        for i in range(1, len(df_prob)):
            neigh_distance = spatial.distance.cosine(df_prob.iloc[i], df_rolling.iloc[i-1])
            neigh_dist_list.append(neigh_distance)
        df_prob = df_prob.reset_index(drop=True)
        df_rolling = df_rolling.reset_index(drop=True)
        df['neighborhood_two_week_cosine_distance'] = neigh_dist_list
        df['entropy_post'] = entropy_list
        #df_prob_check = pd.DataFrame(model.probabilities_)
        df = pd.concat([df, df_prob, df_rolling], axis=1)
        df.to_csv(save_file, index=False)
