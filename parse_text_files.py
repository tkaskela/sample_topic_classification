import pandas as pd
import os

def parse_article_files(directory):
    for file in os.listdir(directory):
        if file.endswith('.txt'):
            print(file)
            file_name = os.path.join(directory, file)
            df = pd.read_table(file_name, header=None)
            df.rename(columns={'Unnamed: 0': 'raw_full'})
            df['date'] = file.rsplit('-', 1)[0]
            df['id'] = df[0].str.split(' ',1).str[0]
            df['text']= df[0].str.split(' ',1).str[1]
            save_name = file_name.split('.')[0] + '_pandas.csv'
            df.to_csv(save_name, index=False)

directory = 'files_to_parse'
parse_article_files(directory)