import pandas as pd

file1 = 'file.csv'
file2 = 'file2.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

df1.drop(['tweet'], inplace=True, axis=1)


df = pd.merge(df1, df2, on='url')
print(df.head())
df.to_csv('file_text_metrics.csv', index=False)
