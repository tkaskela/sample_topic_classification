import pandas as pd
import os

directory = 'test'

for file in os.listdir(directory):
    if file.endswith('.csv'):
        file_name = os.path.join(directory, file)
        df = pd.read_csv(file_name)
        print(df.columns)
        df_brand = df['cleaned_tweet']
        df['max_group'] = df['max_group'].str.split('_').str[1]
        type = df['max_group'].astype(int)
        df = df[['cleaned_tweet', 'max_group']]
        i = 0
        for index, row in df.iterrows():
            if i > len(df):
               break
            else:
               print(type[i])
               brand = df_brand[i]
               strat = type[i]
               f = open(str(strat) + '_tweet_' + str(i) + '.txt', 'w', encoding='utf-8')
               row = row.values.tolist()
               row = str(row)[1:-1]
               f.write(row)
               f.close()
               i+=1