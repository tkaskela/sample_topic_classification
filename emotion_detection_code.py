import pandas as pd
from transformers import pipeline

file = 'file.csv'
column = 'title_self_text_cleaned'
df = pd.read_csv(file)
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, tokenizer="roberta-base")
output = []
n = 1
for index, sentence in df[column].astype(str).items():
    try:
        model_outputs = classifier(sentence)
        model_outputs = sorted(model_outputs[0], key = lambda x: x['label'])
        model_outputs.append({'label': 'index', 'score': index})
        if n == 1:
            output.append([row['label'] for row in model_outputs])
            n += 1
        output.append([row['score'] for row in model_outputs])
    except RuntimeError:
        print('Too long')
    except IndexError:
        print('IndexError: ' + str(index))

df2 = pd.DataFrame(output)
df2.columns = df2.iloc[0]
df2 = df2[1:]
#df2['max_topic'] = df2.idxmax(axis=1)
df2.to_csv('initial_output.csv')
df = pd.merge(df, df2, left_index=True, right_on='index')
save_name = file.split('.')[0] + '_emotion_detection.csv'
df.to_csv(save_name)

