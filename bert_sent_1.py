import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

file = r'file.csv'
df = pd.read_csv(file)

pre_trained_model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(pre_trained_model_name)
sample_txt = 'The refinement and sumptuousness of the Heritage Mahagony Leather enhance the contemporary elegance of ' \
             'this Tailor Made #Ferrari812GTS with an iconic finish, entirely in British Racing Green. ' \
             '#Ferrari #FerrariTailorMade'

tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f' Sentence: {sample_txt}')
print(f'   Tokens: {tokens}')
print(f'Token IDs: {token_ids}')

encoding = tokenizer.encode_plus(
  sample_txt,
  max_length=32,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors
)

token_lens = []
df = df[df['comment_text'].notna()]
for txt in df.comment_text:
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))

sns.distplot(token_lens)
plt.xlim([0, 256])
plt.xlabel('Token count')
plt.show()
