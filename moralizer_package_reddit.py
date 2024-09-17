import pandas as pd
from moralizer import *
from time import time
import os

def moral_foundations_output(directory):
    for file in os.listdir(directory):
        file_name = os.path.join(directory, file)
        start_time = time()
        df = pd.read_csv(file_name)
        print(file)
        column = 'title_self_text'
        df[column] = df['title'].astype(str) + ('_' + df['self_text'].astype(str)).fillna('')
        output = []
        for index, sentence in df[column].astype(str).items():
            print(index)
            model_outputs = moralize(sentence)
            output.append(model_outputs)

        moral_foundations_df = pd.DataFrame(output)
        df = pd.concat([df, moral_foundations_df], axis=1)
        save_file = file.split('.')[0] + '_moral_foundations_added.csv'
        df.to_csv(save_file, index=False)

        end_time = time()

        print('It took ' + str((end_time - start_time)/60) + ' minutes')