import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from model import QGModel
from data_module import QGDataModule
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)


SOURCE_MAX_TOKEN_LEN = 300
TARGET_MAX_TOKEN_LEN = 80
SEP_TOKEN = '<sep>'
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
tokenizer = T5Tokenizer.from_pretrained("./pt_models/t5-small")
# tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
print('tokenizer len before: ', len(tokenizer))
tokenizer.add_tokens(SEP_TOKEN)
print('tokenizer len after: ', len(tokenizer))
TOKENIZER_LEN = len(tokenizer)

data_module = QGDataModule(train_df[:TAKE_TRAIN], dev_df[:TAKE_DEV], test_df[:TAKE_TEST], tokenizer, BATCH_SIZE, SOURCE_MAX_TOKEN_LEN, TARGET_MAX_TOKEN_LEN)
data_module.setup()

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='best-checkpoint',
    save_top_k=-1,
    verbose=True,
    monitor='val_loss',
    mode='min'
)

trainer = pl.Trainer(
    checkpoint_callback=checkpoint_callback,
    max_epochs=N_EPOCHS,
    gpus=1,
    progress_bar_refresh_rate=30
)

model = QGModel()
# model = QGModel.load_from_checkpoint('checkpoints/best-checkpoint-v42.ckpt')

trainer.fit(model, data_module)
