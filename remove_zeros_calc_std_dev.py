import pandas as pd
import numpy as np
from ast import literal_eval
from statistics import stdev
import os


def trim_zeros_calc_std_dev(directory):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(directory, file))
            df['mistral_week_score_trimmed'] = df['mistral_week_score'].apply(lambda x: np.trim_zeros(literal_eval(x)))
            print(df['mistral_week_score_trimmed'])
            df['sd_mistral_weeK_score'] = df['mistral_week_score_trimmed'].apply(lambda x: stdev(x) if len(x) > 1 else 0)
            df.rename(columns={'sd_mistral_weeK_score': 'sd_mistral_week_score'}, inplace=True)
            df['week_count'] = df['mistral_week_score_trimmed'].apply(lambda x: len(x))
            save_name = file.split('.')[0].split('_')[0] + '_final.csv'
            df.to_csv(save_name, index=False)


directory = 'to_run'
trim_zeros_calc_std_dev(directory)
