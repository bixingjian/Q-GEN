from typing import List, Dict
import tqdm.notebook as tq
from tqdm.notebook import tqdm
import json
import pandas as pd
import numpy as np

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)

pl.seed_everything(42)

squad_train_df = pd.read_csv('./dataset/squad1/train_df.csv')
print(squad_train_df.shape)

squad_train_df.head()

squad_dev_df = pd.read_csv('./dataset/squad1/dev_df.csv')
print(squad_dev_df.shape)

squad_dev_df.head()

# Using paragraph
context_name = 'context_para'
drop_context = 'context_sent'

df = squad_train_df.copy()
# print(df.shape, ' :copy')

df = df.dropna() # One missing answer_text. Will fix it later.
# print(df.shape, ' :drop na')

#Dropping duplicates
# df = df.drop_duplicates(subset=['context_sent']).reset_index(drop=True)
# print(df.shape, ' :dropping duplicate sentence')

df.rename(columns = {context_name: 'context'}, inplace=True)
df.drop(columns=[drop_context, 'answer_start', 'answer_end'], inplace=True) #answer_start and answer_end are not needed and are for the paragraph
# print(df.shape, ' :final')

test_df = df[:11877]
train_df = df[11877:]

## Dev set
dev_df = squad_dev_df.copy()
dev_df.rename(columns = {context_name: 'context'}, inplace=True)
dev_df.drop(columns=[drop_context, 'answer_start', 'answer_end'], inplace=True)

print(train_df.shape, 'train_df')
print(dev_df.shape, 'dev_df')
print(test_df.shape, 'test_df')

train_df.head()

print("remote git personal token seems good, this is a test")
print("remote git personal token seems good, this is a test")
print("remote git personal token seems good, this is a test")
