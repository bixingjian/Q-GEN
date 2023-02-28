from typing import List, Dict
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
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
MODEL_NAME = 't5-small'
SOURCE_MAX_TOKEN_LEN = 512
TARGET_MAX_TOKEN_LEN = 64
N_EPOCHS = 20
BATCH_SIZE = 16 #NOTE changed from 24 to 16
LEARNING_RATE = 0.0001
MODEL_SAVE_NAME = '100200'
DF_TAKE_PERCENTAGE = 1



checkpoint_path = 'checkpoints/best-checkpoint-gen-v21.ckpt'

best_model = QGModel.load_from_checkpoint(checkpoint_path)
best_model.freeze()
best_model.eval()

SEP_TOKEN = '<sep>'

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
