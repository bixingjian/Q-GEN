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

checkpoint_path = 'checkpoints/best-checkpoint-v4.ckpt'

best_model = QGModel.load_from_checkpoint(checkpoint_path)
best_model.freeze()
best_model.eval()


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


def show_result(generated: str, answer: str, context:str, original_question: str = ''):
    print('Generated: ', generated)
    if original_question:
        print('Original : ', original_question)

    print()
    print('Answer: ', answer)
    print('Conext: ', context)
    print('-----------------------------')


sample_question = test_df.iloc[42]
generated = generate(best_model, sample_question['answer_text'], sample_question['context'])
show_result(generated, sample_question['answer_text'], sample_question['context'], sample_question['question'])


context = 'Oxygen is the chemical element with the symbol O and atomic number 8.'
answer = 'Oxygen'
generated = generate(best_model, answer, context)
show_result(generated, answer, context)


def show_te_result(te_answer="[MASK]"):
    te_answer = te_answer
    te_generated = generate(best_model, te_answer, te_context)
    show_result(te_generated, te_answer, te_context)


te_context = ''' Perhaps no company embodies the ups and downs of Chinese big tech better than its biggest tech firm of all—Tencent. \
Two years ago the online empire seemed unstoppable. More than a billion Chinese were using its ubiquitous services to pay, play and do much else besides. \
Its video games, such as “League of Legends”, were global hits. \
Tencent’s market value exceeded $900bn, and the firm was on track to become China’s first trillion-dollar company. \
Then the Communist Party said, enough. \
Xi Jinping, China’s paramount leader, decided that big tech’s side-effects, from distracted teenagers to the diversion of capital from strategically important sectors such as semiconductors, were unacceptable. \
Tencent was, along with the rest of China’s once-thriving digital industry, caught up in a sweeping 18-month crackdown. \
'''

show_te_result()
show_te_result()
show_te_result()
show_te_result("Xi Jinping")
show_te_result("Tencent")
show_te_result("the Communist Party")
show_te_result("side-effects")
show_te_result("crackdown")
show_te_result("suppression")

