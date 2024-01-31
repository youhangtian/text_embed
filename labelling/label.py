from llm_api import LLMAPI
from datasets import DatasetDict, Dataset
from tqdm import tqdm

llm_name = 'qwen'
llm_api = LLMAPI(llm_name)

data_dict = {'text': [], 'text_pos': []}

label_file = 'data.txt'
with open(label_file) as f:
    for line in tqdm(f.readlines()):
        line = line.strip()
        prompt = f'作为一名文本分析专家，请设计一个10到20字的问题，让以下内容可以作为这个问题的答案：--- {line} ---。直接说出你设计的问题：'

        try:
            output = llm_api.chat(prompt)

            data_dict['text'].append(output['answer'])
            data_dict['text_pos'].append(line)
        except:
            print('error ------')
            continue 

train_dataset = Dataset.from_dict(data_dict)
my_dataset = DatasetDict({'train': train_dataset})
print(my_dataset)
print(my_dataset['train'][0])

my_dataset.save_to_disk('my_dataset1')
print('done')