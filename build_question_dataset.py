import pandas as pd
from tqdm.notebook import tqdm
from datasets import load_dataset
import pytorch_lightning as pl

pl.seed_everything(42)

def create_dataset(dataset_split): 
    data_rows = []
    for i in tqdm(range(len(dataset_split))):
        cur_context = dataset_split[i]['story_text']
        cur_question = dataset_split[i]['question']

        cur_answer_start_index = int(qa_dataset["train"][0]["answer_token_ranges"].split(":")[0])
        cur_answer_end_inedx = int(qa_dataset["train"][0]["answer_token_ranges"].split(":")[1])
        cur_answer = dataset_split[i]["story_text"].split()[cur_answer_start_index : cur_answer_end_inedx] # in the github issue, the author said the text is separated by space.

        data_rows.append({
            'context': cur_context,
            'question': cur_question,
            'answer_text': cur_answer
        })

    return pd.DataFrame(data_rows)

# get the dataset from huggingface. still need to follow the steps in their github repo.
qa_dataset = load_dataset("newsqa", data_dir="./dataset/newsqa") # in train: features: ['story_id', 'story_text', 'question', 'answer_token_ranges'],num_rows: 92549

question_train_df = create_dataset(qa_dataset['train'])
question_dev_df = create_dataset(qa_dataset['validation'])
question_test_df = create_dataset(qa_dataset['test'])

train_df = question_train_df
dev_df = question_dev_df
test_df = question_test_df
train_df.to_csv('dataset/newsqa/question_train_df.csv', index=False)
dev_df.to_csv('dataset/newsqa/question_dev_df.csv', index=False)
test_df.to_csv('dataset/newsqa/question_test_df.csv', index=False)
