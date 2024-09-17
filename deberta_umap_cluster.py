import umap
import hdbscan
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

file = 'file.csv'
df = pd.read_csv(file)

model = SentenceTransformer('all-mpnet-base-v2')
encoded_text = model.encode(df['tweet'], show_progress_bar=True)

umap_embeddings = umap.UMAP(n_components=5,
                            densmap=True,
                            metric='cosine',
                            n_neighbors=30).fit_transform(encoded_text)

cluster = hdbscan.HDBSCAN(min_cluster_size=35,
                          min_samples=7,
                          metric='euclidean',
                          cluster_selection_method='eom').fit(encoded_text)



result = pd.DataFrame(umap_embeddings, columns=['1', '2', '3', 'x', 'y'])
#result = pd.DataFrame(cluster)
result['labels'] = cluster.labels_

result.to_csv('deberta_cluster.csv')



# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
plt.colorbar()
plt.show()
