import pandas as pd
import ast
import os


def moral_foundations_dictionary_count(directory):
    for file in os.listdir(directory):
        if file.endswith('csv'):
            print(file)
            df = pd.read_csv(os.path.join(directory, file))
            df['care'] = df['care'].apply(lambda x: ast.literal_eval(str(x)))
            df['fairness'] = df['fairness'].apply(lambda x: ast.literal_eval(str(x)))
            df['loyalty'] = df['loyalty'].apply(lambda x: ast.literal_eval(str(x)))
            df['authority'] = df['authority'].apply(lambda x: ast.literal_eval(str(x)))
            df['sanctity'] = df['sanctity'].apply(lambda x: ast.literal_eval(str(x)))

            list_columns = ['care', 'fairness', 'loyalty', 'authority', 'sanctity']

            if 'title_self_text' in df.columns.values:
                column_text = 'title_self_text'
            elif 'text' in df.columns.values:
                column_text = 'text'
            else:
                print('error')

            for moral in list_columns:
                vice = []
                virtue = []
                length = []
                for i in range(len(df[moral])):
                    length.append(len(df[column_text][i].split()))
                    vice.append(sum(df[moral][i]['vice'].values()))
                    virtue.append(sum(df[moral][i]['virtue'].values()))
                df[moral + '_vice'] = vice
                df[moral + '_virtue'] = virtue
                df['title_self_text_length'] = length
                df[moral + '_vice_percentage'] = df[moral + '_vice']/df['title_self_text_length']
                df[moral + '_virtue_percentage'] = df[moral + '_virtue'] / df['title_self_text_length']

            save_name = file.split('.')[0] + '_moral_foundation_count.csv'
            df.to_csv(save_name, index=False)


directory = 'moral_foundations'
moral_foundations_dictionary_count(directory)

