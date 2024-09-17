import pandas as pd
import os


def search_df_csv(directory):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            print(file)
            file_name = os.path.join(directory, file)
            df = pd.read_csv(file_name)
            strings = ['artificial intelligence', 'siri\b', '\bai\b', '\balexa\b', 'chatgpt', 'openai', 'a.i.']
            regstr = '|'.join(strings)
            df = df.loc[df['text'].apply(lambda x: any(word in str(x) for word in strings))]
            save_name = file.split('.')[0] + '_ai_search.csv'
            df.to_csv(save_name, encoding='utf-8', index=False)


directory = r'H:\files_to_parse'
search_df_csv(directory)