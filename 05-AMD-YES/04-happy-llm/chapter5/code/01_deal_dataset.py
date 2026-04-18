import os
import json
from tqdm import tqdm

# pretrain_data is the local path to mobvoi_seq_monkey_general_open_corpus.jsonl downloaded by download_dataset.sh, e.g. './datasets/mobvoi_seq_monkey_general_open_corpus.jsonl'
# pretrain_data 为运行 download_dataset.sh 时，下载的 pretrain_data 本地路径下的 mobvoi_seq_monkey_general_open_corpus.jsonl 文件，例如：'./datasets/mobvoi_seq_monkey_general_open_corpus.jsonl'
pretrain_data = './datasets/mobvoi_seq_monkey_general_open_corpus.jsonl'
output_pretrain_data = './tmp/seq_monkey_datawhale.jsonl'

# sft_data is the local path to train_3.5M_CN.json downloaded by download_dataset.sh, e.g. './datasets/BelleGroup/train_3.5M_CN.json'
# sft_data 为运行 download_dataset.sh 时，下载的 sft_data 本地路径下的 train_3.5M_CN.json 文件，例如：'./datasets/BelleGroup/train_3.5M_CN.json'
sft_data = './datasets/BelleGroup/train_3.5M_CN.json'
output_sft_data = './tmp/BelleGroup_sft.jsonl'

# 1) Process pretraining dataset
# 1 处理预训练数据
def split_text(text, chunk_size=512):
    """Split text into chunks of a specified length.
    将文本按指定长度切分成块
    """
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

with open(output_pretrain_data, 'a', encoding='utf-8') as pretrain:
    with open(pretrain_data, 'r', encoding='utf-8') as f:
        data = f.readlines()
        # Add per-line progress bar
        # 添加行级别的进度条
        for line in tqdm(data, desc=f"Processing lines in {pretrain_data}", leave=False):
            line = json.loads(line)
            text = line['text']
            chunks = split_text(text)
            for chunk in chunks:
                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

# 2) Process SFT dataset
# 2 处理SFT数据
def convert_message(data):
    """
    Convert raw records to the standard message format.
    将原始数据转换为标准格式
    """
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

with open(output_sft_data, 'a', encoding='utf-8') as sft:
    with open(sft_data, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for item in tqdm(data, desc="Processing", unit="lines"):
            item = json.loads(item)
            message = convert_message(item['conversations'])
            sft.write(json.dumps(message, ensure_ascii=False) + '\n')