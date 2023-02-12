import pytorch_lightning as pl
import pandas as pd
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
pl.seed_everything(42)


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


MODEL_NAME = 't5-small'
SOURCE_MAX_TOKEN_LEN = 300
TARGET_MAX_TOKEN_LEN = 80

N_EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

DF_TAKE_PERCENTAGE = 1

train_df = pd.read_csv("./dataset/squad1_preprocessed/train_df.csv")
dev_df = pd.read_csv("./dataset/squad1_preprocessed/dev_df.csv")
test_df = pd.read_csv("./dataset/squad1_preprocessed/test_df.csv")
TAKE_TRAIN = int(len(train_df) * DF_TAKE_PERCENTAGE)
TAKE_DEV = int(len(dev_df) * DF_TAKE_PERCENTAGE)
TAKE_TEST = int(len(test_df) * DF_TAKE_PERCENTAGE)

print('Taking', DF_TAKE_PERCENTAGE * 100, '%')
print(TAKE_TRAIN, 'of', len(train_df))
print(TAKE_DEV, 'of', len(dev_df))
print(TAKE_TEST, 'of', len(test_df))
