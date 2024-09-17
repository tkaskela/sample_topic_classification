from sklearn.cluster import KMeans
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
from umap.umap_ import UMAP

file = 'file.csv'
df = pd.read_csv(file)

model = SentenceTransformer('all-mpnet-base-v2')
text = model.encode(df['tweet'], show_progress_bar=True)

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
    print(sse)
    print(max(sse))
    print(sse.index(max(sse)))
    print(sse.index(max(sse)) + 2)
    return sse.index(max(sse)) + 2

text = UMAP(n_components=5,
                            densmap=True,
                            metric='cosine',
                            n_neighbors=30).fit_transform(text)

optimal_clusters = find_optimal_clusters(text, 20)



def get_top_keywords(data, clusters, labels, n_terms):
    print(pd.DataFrame(data.todense()))
    df = pd.DataFrame(data.todense()).groupby(clusters).mean()
    print(df)
    df_terms = pd.DataFrame()
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(i, r)
        print(','.join([labels[t] for t in np.argsort(r)[-n_terms:]]))
        df_terms[i] = [labels[t] for t in np.argsort(r)[-n_terms:]]
    print(df_terms)
    save_file = 'kmeans_' + str(len(df)) + '_topics_keywords.csv'
    df_terms.to_csv(save_file, index=False)

print(optimal_clusters)
clusters = KMeans(n_clusters=optimal_clusters, random_state=10).fit_predict(text)
df['kmeans_cluster'] = clusters
print(df['kmeans_cluster'])
df.to_csv('file_save_with_cluster.csv', index=False, encoding='utf-8')
