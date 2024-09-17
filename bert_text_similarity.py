from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-mpnet-base-v2')


def similarity_score(directory):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            print(file)
            file_name = os.path.join(directory, file)
            df = pd.read_csv(file_name, dtype={'reply_id': str})
            print(df.columns)
            df = df.sort_values('date')
            score = [0]
            for i in range(1, len(df)):
                str1_int = i + 1
                str2_int = i - 10
                if str2_int < 0:
                    str2_int = 0
                sen = df['tweet'][str2_int:str1_int].to_list()
                print(sen)
                sen_embeddings = model.encode(sen)
                sim_score = cosine_similarity(sen_embeddings[0:-1], [sen_embeddings[-1]])
                score.append(sim_score)
                print(sim_score)

            print(len(df))
            print(len(score))
            df['similarity_bert_score'] = score
            df['mean_bert_score'] = df['similarity_bert_score'].apply(np.mean)
            df['sd_bert_score'] = df['similarity_bert_score'].apply(np.std)
            save_file = df['username'][0] + '_with_bert_10_similarity.csv'
            df.to_csv(save_file, index=False, encoding='utf-8')


def one_prior_similarity(directory):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_name = os.path.join(directory, file)
            df = pd.read_csv(file_name, dtype={'reply_id': str}, encoding='utf-8')
            print(len(df))
            sen_encodings = model.encode(df['tweet'])
            encoding = sen_encodings[0].reshape(-1, 1)
            encoding_2 = sen_encodings[1].reshape(-1, 1)
            sim_score = cosine_similarity(encoding, encoding_2)
            scores = [0]
            for i in range(0, len(df)-1):
                sim_score = cosine_similarity([sen_encodings[i]], [sen_encodings[i+1]])
                scores.append(sim_score[0][0])
            print(scores)
            print(len(scores))
            df['one_row_similarity'] = scores
            df = df[['url', 'one_row_similarity']]
            save_name = file.split('.')[0] + '_only_one_row_similarity_.csv'
            df.to_csv(save_name, index=False)


def same_row_similarity_score(directory):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_name = os.path.join(directory, file)
            print(file)
            df = pd.read_csv(file_name, dtype={'reply_id': str})
            score = []
            sen_embeddings = [model.encode([x, y]) for x, y in zip(df['tweet_x'], df['tweet_y'])]
            for i in range(0, len(sen_embeddings)):
                sim_score = cosine_similarity([sen_embeddings[i][0], sen_embeddings[i][1]])
                score.append(sim_score[0][1])
                print('Score for row ' + str(i) + ' is: ' + str(sim_score[0][1]))
            df['company_reply_bert_score'] = score
            save_file = file.split('.')[0] + '_with_reply_similarity.csv'
            df.to_csv(save_file, index=False, encoding='utf-8')

#directory = 'similarity_to_stack'
#similarity_score(directory)

#directory = 'data_to_stack'
#same_row_similarity_score(directory)

directory = 'data_to_stack'
one_prior_similarity(directory)

