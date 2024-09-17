import pandas as pd
import os
import glob

os.chdir('to_run')
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

print(all_filenames)
combined_csv = pd.concat([pd.read_csv(f, encoding_errors='ignore', sep=',') for f in all_filenames], ignore_index=True, axis=0)
print(len(combined_csv))
print(len(combined_csv))
print(combined_csv.columns)
combined_csv = combined_csv[['url', 'mistral_week_score_trimmed', 'sd_mistral_week_score', 'week_count']]
combined_csv.rename(columns={'week_count': 'week_level_count'})
save_file = all_filenames[0].split('_')[0] + '_week_to_week.csv'
combined_csv.to_csv(save_file, index=False, encoding='utf-8')
print(len(combined_csv))
