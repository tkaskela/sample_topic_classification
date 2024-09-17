import pandas as pd

file = 'file.csv'
df = pd.read_csv(file)
print(df.columns)

std_level_prob = [50, 60, 70, 80, 90, 95, 99]
perc_prob = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99]

for i in range(0, len(perc_prob)):
    col_name = 'hot_perc_streak_' + str(perc_prob[i])
    df['start_of_streak'] = df[col_name].ne(df[col_name].shift())
    df['streak_id'] = df['start_of_streak'].cumsum()
    df['streak_counter'] = df.groupby('streak_id').cumcount() + 1
    df_streak = df[[col_name, 'start_of_streak', 'streak_id', 'streak_counter']]
    print(df_streak.groupby([col_name]).nunique())


for i in range(0, len(std_level_prob)):
    col_name = 'hot_streak_' + str(std_level_prob[i])
    df['start_of_streak'] = df[col_name].ne(df[col_name].shift())
    df['streak_id'] = df['start_of_streak'].cumsum()
    df['streak_counter'] = df.groupby('streak_id').cumcount() + 1
    df_streak = df[[col_name, 'start_of_streak', 'streak_id', 'streak_counter']]
    print(df_streak.groupby([col_name]).nunique())
