import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from .utils import rematch

__all__=['infer']


class _CustomDataset4test(Dataset):
    def __init__(self, data, tokenizer, maxlen, c_size):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.c_size = c_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        label = torch.zeros((self.c_size, self.maxlen, self.maxlen))
        enc_context = self.tokenizer(d[0], return_offsets_mapping=True, max_length=self.maxlen, truncation=True,
                                padding='max_length', return_tensors='pt')
                                
        enc_context = {key: enc_context[key][0] for key in enc_context.keys() if enc_context[key].shape[0] == 1}

        return enc_context

class _NamedEntityRecognizer(object):
    """
    命名实体识别器
    """
    def __init__(self, maxlen, c_size, id2c, device, tokenizer, model_base):
        self.maxlen = maxlen
        self.c_size = c_size
        self.id2c = id2c
        self.device = device
        self.tokenizer = tokenizer
        self.model_base = model_base

    def recognize(self, text, threshold=0):
        
        self.model_base.eval()
        text = text[0:self.maxlen]
 
        tokens = self.tokenizer.tokenize(text)
        mapping = rematch(self.tokenizer, text, tokens)
        text = [[str(text)]]
        data = _CustomDataset4test(text,self.tokenizer,self.maxlen,self.c_size)
        data = DataLoader(data)
        input_ids = []
        attention_mask = []
        token_type_ids = []
        
        for data_1 in data:
            input_ids = data_1['input_ids'].to(self.device)
            attention_mask = data_1['attention_mask'].to(self.device)
            token_type_ids = data_1['token_type_ids'].to(self.device)


        with torch.no_grad():
            scores = self.model_base(input_ids, attention_mask, token_type_ids)

        entities = []
        lenss = len(mapping)

        
        score = scores[0,:,0:lenss,0:lenss].cpu().numpy()
        score = np.triu(score)

  
        for l, start, end in zip(*np.where(score > threshold)):
            entities.append(
                (mapping[start-1][0], mapping[end-1][-1], self.id2c[l])
            )
            
        return entities

    def predict_to_file(self, in_file, out_file):
        """
        预测到文件
        可以提交到 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
        """
        data = json.load(open(in_file))
        for d in data:
            d['entities'] = []
            entities = self.recognize(d['text'])
            print(entities)
            for e in entities:
                d['entities'].append({
                    'start_idx': e[0],
                    'end_idx': e[1],
                    'type': e[2]
                    #'entity': e[3]
                })
        json.dump(
            data,
            open(out_file, 'w', encoding='utf-8'),
            indent=4,
            ensure_ascii=False
        )

_id2c = {0: 'dis', 1: 'sym', 2: 'pro', 3: 'equ', 4: 'dru', 5: 'ite', 6: 'bod', 7: 'dep', 8: 'mic'}

def infer(in_file, out_file='./CMeEE_test_answer.json', 
            model_save_path='./checkpoint/68.2796_macbert_large/macbert-large', 
            maxlen=512, c_size=9, id2c=_id2c):
    '''
    Args:
        in_file (string):
        out_file (string, optional):
        model_save_path (string, optional):
        maxlen (int, optional):
        c_size (int, optional):
        id2c (dictionary, optional):
    '''
    if not os.path.exists(in_file):
        print("[ERROR] in_file does not exist!")
        return
    if not os.path.isfile(in_file):
        print("[ERROR] in_file is not a file!")
        return
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device".format(device))

    model_name = 'hfl/chinese-macbert-large'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if not os.path.exists(model_save_path):
        model_dir = os.path.dirname(model_save_path)
        os.makedirs(model_dir)
        # Reference:
        # https://github.com/pytorch/pytorch/blob/b5b62b340891f041b378681577c74922d21700a9/torch/hub.py
        url = 'https://github.com/cb1cyf/NNER/releases/download/v0.0.1/macbert-large'
        try:
            model_base = torch.hub.load_state_dict_from_url(url, model_dir)
        except:
            torch.hub.download_url_to_file(url, model_save_path)
            model_base = torch.load(model_save_path)
    else:
        model_base = torch.load(model_save_path)
    NER = _NamedEntityRecognizer(maxlen, c_size, id2c, device, tokenizer, model_base)
    NER.predict_to_file(in_file, out_file)



if __name__ == '__main__':
    in_file = './dataset/normal/CMeEE_test.json'
    infer(in_file)