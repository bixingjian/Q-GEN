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

train_df = pd.read_csv('dataset/race/race_test_df.csv')
print(train_df["context"][0])

# from datasets import load_dataset

# dataset = load_dataset("race", 'all')

# print(dataset['train']['options'][0])
