import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
from sentence_transformers import util
import os
import statistics
from statistics import stdev

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

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


def semantic_similarity_periods(directory, days=7):
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df['start_date'] = pd.to_datetime(df['date'])
            df['end_date'] = df['start_date'] - pd.DateOffset(days=days)
            df = df.sort_values(by='date', ascending=False)
            mistral_all = []
            sd_mis_all = []
            for idx, row in df.iterrows():
                mistral_score = []
                sd_mis = []
                end_date = row['start_date']
                start_date = row['start_date'] - pd.DateOffset(days=days)
                print(start_date, end_date)
                primary_df = df.loc[df['date'].between(start_date, end_date)]
                print(primary_df)
                if len(primary_df) > 0:
                    primary_text = ', '.join(primary_df['tweet'])
                else:
                    primary_text = ' '
                print(primary_text)
                for i, r in df.iloc[1:].iterrows():
                    print(r['tweet'])
                    new_start = start_date - pd.DateOffset(days=7)
                    new_end = new_start - pd.DateOffset(days=7)
                    print(new_end)
                    print(new_start)
                    secondary_df = df.loc[df['date'].between(new_end, new_start)]
                    print(len(secondary_df))
                    if len(secondary_df) > 0:
                        second_text = ', '.join(secondary_df['tweet'])
                        encoded_input = tokenizer([primary_text, second_text], padding=True, truncation=True, return_tensors='pt')
                        with torch.no_grad():
                            model_output = model(**encoded_input)

                        # Perform pooling
                        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

                        # Normalize embeddings
                        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                        cosine_scores = util.cos_sim(sentence_embeddings[0], sentence_embeddings[1])
                        print(cosine_scores[0][0].item())
                        mistral_score.append(cosine_scores[0][0].item())
                    else:
                        mistral_score.append(0)
                    start_date = new_start

                mistral_all.append(mistral_score)
                try:
                    sd_mis_all.append(stdev(mistral_score))
                except (statistics.StatisticsError, ValueError):
                    sd_mis_all.append(0)
            print(sd_mis_all)
            print(mistral_all)
            print(len(sd_mis_all))
            print(len(mistral_all))
            df['sd_mistral_week_score'] = sd_mis_all
            df['mistral_week_score'] = mistral_all
            df.to_csv(file.split('.')[0] + '_one_week_comparison.csv', index=False)


directory = 'week_to_week_similarity'
semantic_similarity_periods(directory, 7)