import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from question_model import QGModel
from build_question_data_module import QGDataModule
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)

Q_SOURCE_MAX_TOKEN_LEN = 300
Q_TARGET_MAX_TOKEN_LEN = 80
SEP_TOKEN = '<sep>'

q_tokenizer = T5Tokenizer.from_pretrained("./pt_models/t5-small")
# tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
print('tokenizer len before: ', len(q_tokenizer))
q_tokenizer.add_tokens(SEP_TOKEN)
print('tokenizer len after: ', len(q_tokenizer))
TOKENIZER_LEN = len(q_tokenizer)

checkpoint_path = 'checkpoints/best-checkpoint-v4.ckpt'

best_model = QGModel.load_from_checkpoint(checkpoint_path)
best_model.freeze()
best_model.eval()


def generate(qgmodel: QGModel, answer: str, context: str) -> str:
    source_encoding = q_tokenizer(
        '{} {} {}'.format(answer, SEP_TOKEN, context),
        max_length=Q_SOURCE_MAX_TOKEN_LEN,
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
        max_length=Q_TARGET_MAX_TOKEN_LEN,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        use_cache=True
    )

    preds = {
        q_tokenizer.decode(generated_id, skip_special_tokens=False, clean_up_tokenization_spaces=True)
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


