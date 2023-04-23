from typing import List, Dict
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from model import QGModel # just write the class, cause it contains a TOKENIZER_LEN needed to be set.
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)
from context_list import *

class QGModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
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
    
    
print("--------")

SEP_TOKEN = '<sep>'
SEP1 = '<distractor_1>'
SEP2 = '<distractor_2>'
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
print('tokenizer len before: ', len(tokenizer))
tokenizer.add_tokens([SEP_TOKEN, SEP1, SEP2])
print('tokenizer len after: ', len(tokenizer))
TOKENIZER_LEN = len(tokenizer)

SOURCE_MAX_TOKEN_LEN = 512
TARGET_MAX_TOKEN_LEN = 64
N_EPOCHS = 20
BATCH_SIZE = 16 #NOTE changed from 24 to 16
LEARNING_RATE = 0.0001
MODEL_SAVE_NAME = '100200'
DF_TAKE_PERCENTAGE = 1

print("--------")



checkpoint_path = 'checkpoints_distractor_sep_1_2/checkpoint-v15.ckpt'

best_model = QGModel.load_from_checkpoint(checkpoint_path)
best_model.freeze()
best_model.eval()


print("--------")

def generate(qgmodel: QGModel, answer: str, context: str) -> str:
    source_encoding = tokenizer(
        '{} {} {}'.format(answer, SEP_TOKEN, context),
        max_length=SOURCE_MAX_TOKEN_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors='pt'
    )

    generated_ids = qgmodel.model.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask=source_encoding['attention_mask'],
        num_beams=1,
        max_length=TARGET_MAX_TOKEN_LEN,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    preds = {
        tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        for generated_id in generated_ids
    }

    return ''.join(preds)

def show_result(generated: str, answer: str, context:str, incorrect: List[str] = [], question: str = ''):
    print('Context:')
    print(context)
    print()

    if question: print('Question: ', question)
    print('Answer : ', answer)

    print()
    print('Original : ', incorrect)
    print('Generated: ', generated)
    print('-----------------------------')


print("--------")
test_df = pd.read_csv("./dataset/race/race_test_df.csv")
sample = test_df.iloc[42]


generated_distractor_with_special = []
for i in range(10):
    generated = generate(best_model, correct_answer_list[i], context_list[i])
    generated_distractor_with_special.append(generated)
    show_result(generated, correct_answer_list[i], context_list[i], [test_df.iloc[i][3], test_df.iloc[i][4], test_df.iloc[i][5]])

print(generated_distractor_with_special)

generated_distractor = [] # without sepcial tokens
for i in range(len(generated_distractor_with_special)):
    temp = []
    s = generated_distractor_with_special[i]
    delimiter = '<pad> '
    substrings = s.split(delimiter)
    substring_1 = substrings[1].split('<distractor_1> ')[0].rstrip()
    substring_2 = substrings[1].split('<distractor_2> ')[0].split('<distractor_1>')[1].rstrip()
    substring_3 = substrings[1].split('</s>')[0].split('<distractor_2>')[1].rstrip()
    temp.append(substring_1)
    temp.append(substring_2)
    temp.append(substring_3)
    generated_distractor.append(temp)

print(generated_distractor)

# test_context = ''' Perhaps no company embodies the ups and downs of Chinese big tech better than its biggest tech firm of all—Tencent. \
# Two years ago the online empire seemed unstoppable. More than a billion Chinese were using its ubiquitous services to pay, play and do much else besides. \
# Its video games, such as “League of Legends”, were global hits. \
# Tencent’s market value exceeded $900bn, and the firm was on track to become China’s first trillion-dollar company. \
# Then the Communist Party said, enough. \
# Xi Jinping, China’s paramount leader, decided that big tech’s side-effects, from distracted teenagers to the diversion of capital from strategically important sectors such as semiconductors, were unacceptable. \
# Tencent was, along with the rest of China’s once-thriving digital industry, caught up in a sweeping 18-month crackdown. \
# '''

# # test_context = context_list[2]

# test_correct_answer = "distracting teenagers, and causing diversion of capital"

