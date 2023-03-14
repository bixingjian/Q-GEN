import numpy as np
import pytorch_lightning as pl
import pandas as pd
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader

pl.seed_everything(42)


PT_MODEL_PATH = "./pt_models/t5-base"
SOURCE_MAX_TOKEN_LEN = 300
TARGET_MAX_TOKEN_LEN = 80
MASKING_CHANCE = 0.3
SEP_TOKEN = '<sep>' # not using SEP, caused that is already a token in the vocabulary
BATCH_SIZE = 16

tokenizer = T5Tokenizer.from_pretrained(PT_MODEL_PATH)
print('tokenizer len before: ', len(tokenizer))
tokenizer.add_tokens(SEP_TOKEN)
print('tokenizer len after: ', len(tokenizer))
TOKENIZER_LEN = len(tokenizer)


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
        self.train_dataset = QGDataset(self.train_df, self.tokenizer, self.source_max_token_len,
                                       self.target_max_token_len)
        self.val_dataset = QGDataset(self.val_df, self.tokenizer, self.source_max_token_len, self.target_max_token_len)
        self.test_dataset = QGDataset(self.test_df, self.tokenizer, self.source_max_token_len,
                                      self.target_max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=2)


train_df = pd.read_csv("./dataset/squad1_preprocessed/train_df.csv")
dev_df = pd.read_csv("./dataset/squad1_preprocessed/dev_df.csv")
test_df = pd.read_csv("./dataset/squad1_preprocessed/test_df.csv")
print("train df shape {}, dev df shape {}, test df shape {}".format(train_df.shape, dev_df.shape, test_df.shape))


data_module = QGDataModule(train_df, dev_df, test_df, tokenizer, BATCH_SIZE, SOURCE_MAX_TOKEN_LEN, TARGET_MAX_TOKEN_LEN)
data_module.setup()
