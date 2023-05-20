import pandas as pd
from question_model import QGModel
from transformers import T5TokenizerFast as T5Tokenizer
from context_list import context_list

SOURCE_MAX_TOKEN_LEN = 1024
TARGET_MAX_TOKEN_LEN = 80
SEP_TOKEN = '<sep>'
PT_MODEL_PATH = "./pt_models/t5-small"
train_df = pd.read_csv("./dataset/squad1_preprocessed/train_df.csv")
dev_df = pd.read_csv("./dataset/squad1_preprocessed/dev_df.csv")
test_df = pd.read_csv("./dataset/squad1_preprocessed/test_df.csv")
tokenizer = T5Tokenizer.from_pretrained(PT_MODEL_PATH)
tokenizer.add_tokens(SEP_TOKEN)
TOKENIZER_LEN = len(tokenizer)

checkpoint_path = 'checkpoints_question/best-checkpoint-v8.ckpt'

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


# sample_question = test_df.iloc[42]
# generated = generate(best_model, sample_question['answer_text'], sample_question['context'])
# show_result(generated, sample_question['answer_text'], sample_question['context'], sample_question['question'])


# context = 'Oxygen is the chemical element with the symbol O and atomic number 8.'
# answer = 'Oxygen'
# generated = generate(best_model, answer, context)
# show_result(generated, answer, context)


def show_test_result(test_answer="[MASK]"):
    test_answer = test_answer
    te_generated = generate(best_model, test_answer, test_context)
    show_result(te_generated, test_answer, test_context)


test_context = ''' Perhaps no company embodies the ups and downs of Chinese big tech better than its biggest tech firm of all — Tencent. \
Two years ago the online empire seemed unstoppable. More than a billion Chinese were using its ubiquitous services to pay, play and do much else besides. \
Its video games, such as “League of Legends”, were global hits. \
Tencent’s market value exceeded $900bn, and the firm was on track to become China’s first trillion-dollar company. \
Chinese leaders decided that big tech’s side-effects, from distracted teenagers to the diversion of capital from strategically important sectors such as semiconductors, were unacceptable. \
Tencent was, along with the rest of China’s once-thriving digital industry, caught up in a sweeping 18-month crackdown. \
'''

# test_context = context_list[2]
# print(len(test_context))



show_test_result()
# show_test_result("Xi Jinping")
# show_test_result("Tencent")
show_test_result("distracting teenagers")
show_test_result("$900bn")
# show_test_result("crackdown")
# show_test_result("suppression")
show_test_result("online services")
# show_test_result("Protect her cows from being drowned")
# show_test_result("distracting teenagers, and causing diversion of capital")

