import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer, util
from statistics import stdev
import statistics


'''tokenizer = AutoTokenizer.from_pretrained('Salesforce/SFR-Embedding-Mistral')
model = AutoModel.from_pretrained('Salesforce/SFR-Embedding-Mistral')'''

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

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


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def run_rolling_window_similarity(directory):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            print(file)
            df = pd.read_csv(os.path.join(directory, file))
            print(df)
            print(df.columns)
            print(df['date'])
            df['date'] = pd.to_datetime(df['date'])
            print(type(df['date'][0]))
            print(df['date'])

            df = df.sort_values(by='date', ascending=False)
            #df = df.sort_values(by='date', ascending=True)
            df['one_week_rolling_count_reverse'] = df.rolling(on='date', window='7D', min_periods=1).count()['Followers']
            df.to_csv('test_test_rolling.csv', index=False)
            df['tweet'] = df['tweet'].fillna(' ')
            df = df.sort_values(by='date', ascending=True).reset_index(drop=True)
            fuzz_all = []
            mistral_all = []
            sd_mis_all = []
            sd_fuzz_all = []
            for index, row in df.iterrows():
                fuzz_score = [0]
                mistral_score = [0]
                print(index)
                print(row['one_week_rolling_count_reverse'])
                for calc_row in range(1, int(row['one_week_rolling_count_reverse'])):
                    print(index)
                    print(index + calc_row)
                    str1 = df.loc[index]['tweet']
                    try:
                        str2 = df.loc[index + calc_row]['tweet']
                    except KeyError:
                        str2 = ' '
                    encoded_input = tokenizer([str1, str2], padding=True, truncation=True, return_tensors='pt')
                    with torch.no_grad():
                        model_output = model(**encoded_input)

                    # Perform pooling
                    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

                    # Normalize embeddings
                    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                    cosine_scores = util.cos_sim(sentence_embeddings, sentence_embeddings)
                    print(cosine_scores[0][1].item())
                    mistral_score.append(cosine_scores[0][1].item())
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

                fuzz_all.append(fuzz_score)
                try:
                    sd_fuzz_all.append(stdev(fuzz_score))
                except statistics.StatisticsError:
                    sd_fuzz_all.append(0)
                mistral_all.append(mistral_score)
                try:
                    sd_mis_all.append(stdev(mistral_score))
                except statistics.StatisticsError:
                    sd_mis_all.append(0)
                #print(mistral_all)

            df['mistral_similarity_prior'] = mistral_all
            df['fuzz_similarity_prior'] = fuzz_all
            df['sd_mistral_similarity_prior'] = sd_mis_all
            df['sd_fuzz_similarity_prior'] = sd_fuzz_all
            df.to_csv(file.split('.')[0] + '_similarity_score_reversed.csv', index=False)


directory = 'simple_fuzzy_nasdaq_similarity_completed'
run_rolling_window_similarity(directory)
