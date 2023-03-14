import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from question_model import QGModel
from question_data_module import QGDataModule
from transformers import T5TokenizerFast

pl.seed_everything(42)


SEP_TOKEN = '<sep>'
PT_MODEL_PATH = "./pt_models/t5-small"
SOURCE_MAX_TOKEN_LEN = 300
TARGET_MAX_TOKEN_LEN = 80
N_EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

train_df = pd.read_csv("./dataset/newsqa/question_train_df.csv")
dev_df = pd.read_csv("./dataset/newsqa/question_dev_df.csv")
test_df = pd.read_csv("./dataset/newsqa/question_test_df.csv")
print("train df shape {}, dev df shape {}, test df shape {}".format(train_df.shape, dev_df.shape, test_df.shape))

tokenizer = T5TokenizerFast.from_pretrained(PT_MODEL_PATH)
print('tokenizer len before: ', len(tokenizer))
tokenizer.add_tokens(SEP_TOKEN)
print('tokenizer len after: ', len(tokenizer))
TOKENIZER_LEN = len(tokenizer)


data_module = QGDataModule(train_df, dev_df, test_df, tokenizer, BATCH_SIZE, SOURCE_MAX_TOKEN_LEN, TARGET_MAX_TOKEN_LEN)
data_module.setup()


checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints_question',
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
trainer.fit(model, data_module)
trainer.test()
