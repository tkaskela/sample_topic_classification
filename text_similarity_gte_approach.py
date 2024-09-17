from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd
import os

model = SentenceTransformer('Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)

def text_similarity_gte(directory, search_term):
    for file in sorted(os.listdir(directory), reverse=True):
        if file.endswith('.csv'):
            print(file)
            df = pd.read_csv(os.path.join(directory, file))
            similarity = []

            for i, row in df.iterrows():
                embeddings = model.encode([search_term, row['text']])
                print(row['text'])
                print(cos_sim(embeddings[0], embeddings[1]).item())
                similarity.append(cos_sim(embeddings[0], embeddings[1]).item())
            df['similarity'] = similarity
            save_file = file.split('.')[0] + '_added_similarity.csv'
            df.to_csv(save_file, index=False)


directory = 'text_similarity_files'
search_term = ['Artificial Intelligence']
text_similarity_gte(directory, search_term)