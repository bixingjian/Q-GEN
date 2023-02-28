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

from datasets import load_dataset

dataset = load_dataset("race", 'all')

print("train.............", dataset['train']['article'][0])

def create_dataset(dataset_split):
    data_rows = []
    for i in tqdm(range(len(dataset_split))):
        curr_context = dataset_split[i]['article']
        curr_question = dataset_split[i]['question']

        all_answers = dataset_split[i]['options']
        correct_answer_index =  ord(dataset_split[i]['answer']) - 65

        curr_correct = all_answers.pop(correct_answer_index)
        curr_incorrect1 = all_answers[0]
        curr_incorrect2 = all_answers[1]
        curr_incorrect3 = all_answers[2]

        data_rows.append({
            'context': curr_context,
            'question': curr_question,
            'correct': curr_correct,
            'incorrect1': curr_incorrect1,
            'incorrect2': curr_incorrect2,
            'incorrect3': curr_incorrect3
        })

    return pd.DataFrame(data_rows)

race_train_df = create_dataset(dataset['train'])
race_dev_df = create_dataset(dataset['validation'])
race_test_df = create_dataset(dataset['test'])
print(race_train_df.head())

train_df = race_train_df
dev_df = race_dev_df
test_df = race_test_df

train_df.to_csv('dataset/race/race_train_df.csv', index=False)
dev_df.to_csv('dataset/race/race_dev_df.csv', index=False)
test_df.to_csv('dataset/race/race_test_df.csv', index=False)

model_name = 't5-small'

tokenizer = T5Tokenizer.from_pretrained(model_name)

train_df = pd.read_csv('dataset/race/race_train_df.csv')
ev_df = pd.read_csv('dataset/race/race_dev_df.csv')
test_df = pd.read_csv('dataset/race/race_test_df.csv')

SEP_TOKEN = '<sep>'

class QGDataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        source_max_token_len: int,
        target_max_token_len: int
        ):

        self.tokenizer = tokenizer
        self.data = data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        source_encoding = tokenizer(
            '{} {} {} {} {}'.format(data_row['correct'], SEP_TOKEN, data_row['question'], SEP_TOKEN, data_row['context']),
            max_length= self.source_max_token_len,
            padding='max_length',
            truncation= True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
            )
    
        target_encoding = tokenizer(
            '{} {} {} {} {}'.format(data_row['incorrect1'], SEP_TOKEN, data_row['incorrect2'], SEP_TOKEN, data_row['incorrect3']),
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation = True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
            )

        labels = target_encoding['input_ids']  
        labels[labels == 0] = -100

        return dict(
            answer_text = data_row['correct'],
            context = data_row['context'],
            question = data_row['question'],
            incorrect1 = data_row['incorrect1'],
            incorrect2 = data_row['incorrect2'],
            incorrect3 = data_row['incorrect3'],
            input_ids = source_encoding['input_ids'].flatten(),
            attention_mask = source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
            )
    
class QGDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size,
        source_max_token_len: int,
        target_max_token_len: int
        ): 
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def setup(self):
        self.train_dataset = QGDataset(self.train_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)
        self.val_dataset = QGDataset(self.val_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)
        self.test_dataset = QGDataset(self.test_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, num_workers = 2)

    def val_dataloader(self): 
        return DataLoader(self.val_dataset, batch_size=1, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=2)
    
MODEL_NAME = 't5-small'
SOURCE_MAX_TOKEN_LEN = 512
TARGET_MAX_TOKEN_LEN = 64

N_EPOCHS = 20
BATCH_SIZE = 20 #NOTE changed from 24 to 16
LEARNING_RATE = 0.0001

MODEL_SAVE_NAME = '100200'

DF_TAKE_PERCENTAGE = 1

TAKE_TRAIN = int(len(train_df) * DF_TAKE_PERCENTAGE)
TAKE_DEV = int(len(dev_df) * DF_TAKE_PERCENTAGE)
TAKE_TEST = int(len(test_df) * DF_TAKE_PERCENTAGE)

print(train_df[:TAKE_TRAIN].shape, dev_df[:TAKE_DEV].shape, test_df[:TAKE_TEST].shape)

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
print('tokenizer len before: ', len(tokenizer))
tokenizer.add_tokens(SEP_TOKEN)
print('tokenizer len after: ', len(tokenizer))
TOKENIZER_LEN = len(tokenizer)

data_module = QGDataModule(train_df[:TAKE_TRAIN], dev_df[:TAKE_DEV], test_df[:TAKE_TEST], tokenizer, BATCH_SIZE, SOURCE_MAX_TOKEN_LEN, TARGET_MAX_TOKEN_LEN)
data_module.setup()

class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        self.model.resize_token_embeddings(TOKENIZER_LEN) #resizing after adding new tokens to the tokenizer

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss, output = self(input_ids, attention_mask, labels)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss
  
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=LEARNING_RATE)
    
checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-checkpoint-gen',
        save_top_k=-1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

trainer = pl.Trainer(
        checkpoint_callback= checkpoint_callback,
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30
    )

model = QGModel()
trainer.fit(model, data_module)
trainer.test()

# checkpoint_path = 'checkpoints/best-checkpoint-v9.ckpt'

# best_model = QGModel.load_from_checkpoint(checkpoint_path)
# best_model.freeze()
# best_model.eval()

# print()

# SEP_TOKEN
