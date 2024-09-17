import pandas as pd
import os


def cluster_change(directory):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            score = [0]
            for i in range(1, len(df)):
                str1_int = i - 1
                str2_int = i
                str1 = df['bert_cluster_assignment'][str1_int]
                str2 = df['bert_cluster_assignment'][str2_int]
                topic_difference = 0 if str1 == str2 else 1
                score.append(topic_difference)
                print(topic_difference)

            print(len(df))
            print(len(score))
            df['cluster_change'] = score
            save_file = df['username'][0] + '_cluster_change.csv'
            df.to_csv(save_file, index=False)


directory = 'text_clusters'
cluster_change(directory)
