import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
from fuzzywuzzy import fuzz


tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')


# get the embeddings
def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def text_similarity_measurement(directory, column_name):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            print(file)
            # Each query must come with a one-sentence instruction that describes the task
            df = pd.read_csv(os.path.join(directory, file))
            print(df.columns)
            df = df.sort_values('date')
            fuzz_score = [0]
            mistral_score = [0]
            for i in range(1, len(df)):
                str1_int = i - 1
                str2_int = i
                str1 = df[column_name][str1_int]
                str2 = df[column_name][str2_int]

                print(df['username'][str1_int])
                print(df['username'][str2_int])

                print(str1)
                print(str2)
                task = 'Given a number of descriptive terms, measure the similarity of each passage with the query'
                queries = [
                    get_detailed_instruct(task,
                                          str1),
                    get_detailed_instruct(task, 'IT, information technology')
                ]

                token_set_ratio = fuzz.token_set_ratio(str1, str2)
                fuzz_score.append(token_set_ratio)
                input_texts = queries + [str2]
                # load model and tokenizer
                max_length = 4096
                try:
                    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
                    outputs = model(**batch_dict)
                    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                    scores = (embeddings[:2] @ embeddings[2:].T) * 100
                    mistral_score.append(scores.tolist()[0][0])
                except:
                    print('Error - Could not embed')
                    mistral_score.append('error')
                    continue

            df['mistral_similarity'] = mistral_score
            df['fuzz_similarity'] = fuzz_score
            df.to_csv(file.split('.')[0] + '_similarity_score.csv', index=False)



directory = 'nasdaq_tweets'
column_name = 'tweet'
text_similarity_measurement(directory, column_name)