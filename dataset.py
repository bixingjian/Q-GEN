from typing import List, Dict
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)

SEP_TOKEN = '<sep>'
MASKING_CHANCE = 0.3  # 30% chance to replace the answer with '[MASK]'


class QGDataset(Dataset):
    def __init__(self,
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

        if np.random.rand() > MASKING_CHANCE:
            answer = data_row['answer_text']
        else:
            answer = '[MASK]'

        source_encoding = tokenizer(
            '{} {} {}'.format(answer, SEP_TOKEN, data_row['context']),
            max_length=self.source_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        target_encoding = tokenizer(
            '{} {} {}'.format(data_row['answer_text'], SEP_TOKEN, data_row['question']),
            max_length=self.target_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = target_encoding['input_ids']
        labels[labels == 0] = -100

        return dict(
            answer_text=data_row['answer_text'],
            context=data_row['context'],
            question=data_row['question'],
            input_ids=source_encoding['input_ids'].flatten(),
            attention_mask=source_encoding['attention_mask'].flatten(),
            labels=labels.flatten()
        )
