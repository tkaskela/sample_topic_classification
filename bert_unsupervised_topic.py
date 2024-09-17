import sklearn.metrics
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

embedder = SentenceTransformer('all-mpnet-base-v2')

file = 'file.csv'
df = pd.read_csv(file)
print(len(df))
column = 'tweet'
df['cleaned_tweet'] = df[column]
df['original_tweet'] = df['cleaned_tweet']
df['cleaned_tweet'] = df['cleaned_tweet'].str.replace(r'\s*https?://\S+(\s+|$)', ' ', regex=True).str.strip()
df['cleaned_tweet'] = df['cleaned_tweet'].str.replace('(\@\w+.*?)',"", regex=True)

df['cleaned_tweet'] = df['cleaned_tweet'].str.lower()
stop = stopwords.words('english')
new_words = ['word', 'word2']
stop.extend(new_words)
df = df[~df['cleaned_tweet'].str.isnumeric().fillna(True)]

#df.to_csv('string_only.csv')
df['cleaned_tweet'] = df[column].str.lower().apply(lambda x: ' '.join([word for word in x.split() if word not in stop]))
print(len(df['cleaned_tweet']))
df.columns = df.columns.str.replace('\r', '')
print(df.columns)
corpus = df['cleaned_tweet']
corpus.reset_index(drop=True, inplace=True)
print(len(corpus))
print('Tweets read, starting encoding embeddings.')
corpus_embeddings = embedder.encode(corpus)


Sum_of_squared_distances = []
silhouetted_score = []
K = range(1,25)
diff_in_diff_list = [0, 0]
diff_sum_list = [0]
diff_in_diff_in_diff_list = [0, 0, 0]
for k in K:
    print('Round: ' + str(k))
    km = KMeans(n_clusters=k)
    km = km.fit(corpus_embeddings)
    Sum_of_squared_distances.append(km.inertia_)
    k_2 = k - 1
    if k != 1:
        silhouetted_score.append(sklearn.metrics.silhouette_score(corpus_embeddings, km.labels_, metric='euclidean'))
        diff_sum = (Sum_of_squared_distances[-2] - Sum_of_squared_distances[-1])/ Sum_of_squared_distances[-2]
        print('Difference is: ' + str(diff_sum))
        diff_sum_list.append(diff_sum)
        print(diff_sum_list)
        if len(diff_sum_list) > 2:
            print('Difference in differences is: ' + str(diff_sum_list[-2] - diff_sum_list[-1]))
            diff_in_diff_list.append(diff_sum_list[-2] - diff_sum_list[-1])
            print(diff_in_diff_list)
            if len(diff_in_diff_list) > 2:
                diff_in_diff_in_diff_list.append(diff_in_diff_list[-2] - diff_in_diff_list[-1])
                print(diff_in_diff_in_diff_list)
                print(diff_in_diff_in_diff_list.index(max(diff_in_diff_in_diff_list, key=abs)) + 1)
    else:
        print('Difference is: 0')


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show(block=True)

clusters = diff_in_diff_in_diff_list.index(max(diff_in_diff_in_diff_list))


def find_optimal_clusters(data, max_k):
    iters = range(2, max_k + 1, 1)
    sse = []
    for k in iters:
        sse.append(KMeans(n_clusters=k, random_state=10).fit(data).inertia_)
        print('Fit {} clusters'.format(k))
    f, ax = plt.subplots(1, 1)
    ax.plot(iters, sse, marker='o')
    ax.set_xlabel('Cluster Centers')
    ax.set_xticks(iters)
    ax.set_xticklabels(iters)
    ax.set_ylabel('SSE')
    ax.set_title('SSE by Cluster Center Plot')
    plt.show()
    optimal_clusters = sse.index(max(sse)) + 2
    return optimal_clusters

optimal_clusters = find_optimal_clusters(corpus_embeddings, 250)

clustering_model = KMeans(n_clusters=optimal_clusters)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_
print(len(cluster_assignment))
clustered_sentences = [[] for i in range(100)]

for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

print('Sentence ID is: ' + str(sentence_id))
df['bert_cluster_assignment'] = cluster_assignment.tolist()
save_name = file.split('.')[0] + '_' + str(40) + '_clusters_newest.csv'
df.to_csv(save_name, index=False)

