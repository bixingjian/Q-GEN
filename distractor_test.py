from typing import List, Dict
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from distractor_gen import QGModel # 要从dis里面导入 因为加入了两个多的token
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)

# class QGModel(pl.LightningModule):
#     def __init__(self):
#         super().__init__()
#         self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
#         self.model.resize_token_embeddings(TOKENIZER_LEN) #resizing after adding new tokens to the tokenizer

#     def forward(self, input_ids, attention_mask, labels=None):
#         output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         return output.loss, output.logits

#     def training_step(self, batch, batch_idx):
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['labels']
#         loss, output = self(input_ids, attention_mask, labels)
#         self.log('train_loss', loss, prog_bar=True, logger=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['labels']
#         loss, output = self(input_ids, attention_mask, labels)
#         self.log('val_loss', loss, prog_bar=True, logger=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['labels']
#         loss, output = self(input_ids, attention_mask, labels)
#         self.log('test_loss', loss, prog_bar=True, logger=True)
#         return loss
  
#     def configure_optimizers(self):
#         return AdamW(self.parameters(), lr=LEARNING_RATE)
    
    
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



checkpoint_path = 'checkpoints-sep_1_2/checkpoint-v15.ckpt'

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
print(sample['context'])

generated = generate(best_model, sample['correct'], sample['context'])
show_result(generated, sample['correct'], sample['context'], [sample['incorrect1'], sample['incorrect2'], sample['incorrect3']], sample['question'])
