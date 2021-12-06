import os
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast
from .model import GlobalPointer
from .utils import rematch

__all__=['load_NNER']


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

class _GlobalPointerNet(nn.Module):
    def __init__(self, model_name, c_size=9, head_size=64, embedding_size=1024):
        super(_GlobalPointerNet, self).__init__()
        self.head = GlobalPointer(c_size, head_size, embedding_size)
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x2 = x1.last_hidden_state
        #last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        # – Sequence of hidden-states at the output of the last layer of the model.

        logits = self.head(x2, mask=attention_mask)
        return logits

class _NamedEntityRecognizer(object):
    """
    Named Entity Recognizer
    """
    def __init__(self, maxlen, c_size, id2c, device, tokenizer, model_base):
        self.maxlen = maxlen
        self.c_size = c_size
        self.id2c = id2c
        self.device = device
        self.tokenizer = tokenizer
        self.model_base = model_base

    def recognize(self, text, threshold=0):
        """
        Args:
            text (str): input sentence
            threshold (int): threshold score to filter out recognized entity with low confidence score
                Default: 0
        Returns:
            entities (list): list of tuples with recognized entity. The format is [(start_id, end_id, entity_class), ...]
        """
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
        Args:
            in_file (str): path of input json file
            out_file (str): path of output json file which can be submitted to CBLUE
        """
        data = json.load(open(in_file))
        for d in data:
            d['entities'] = []
            entities = self.recognize(d['text'])
            #print(entities)
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

def load_NNER(model_save_path='./checkpoint/macbert-large_dict.pth', 
                maxlen=512, c_size=9, id2c=_id2c):
    '''
    Args:
        model_save_path (string, optional): path of pretrained model
            Default: './checkpoint/macbert-large_dict.pth'
        maxlen (int, optional): max length of sentence
            Default: 512
        c_size (int, optional): number of entity class
            Default: 9
        id2c (dictionary, optional): mapping between id and entity class
            Default: _id2c
    Returns:
        NNER (class _NamedEntityRecognizer): the pretrained NNER model
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {} device".format(device))

    model_name = 'hfl/chinese-macbert-large'
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model_base = _GlobalPointerNet(model_name).to(device)

    if not os.path.exists(model_save_path):
        model_dir = os.path.dirname(model_save_path)
        os.makedirs(model_dir)
        # Reference:
        # https://github.com/pytorch/pytorch/blob/b5b62b340891f041b378681577c74922d21700a9/torch/hub.py
        url = 'https://github.com/cb1cyf/NNER/releases/download/v0.0.1/macbert-large_dict.pth'
        #model_base = torch.hub.load_state_dict_from_url(url, model_dir)
        torch.hub.download_url_to_file(url, model_save_path)
    state_dict = torch.load(model_save_path, map_location=device)
    # multi-gpu -> 1 gpu
    model_base.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    NNER = _NamedEntityRecognizer(maxlen, c_size, id2c, device, tokenizer, model_base)
    return NNER


if __name__ == '__main__':
    in_file = '../CMeEE_test.json'
    #infer(in_file)

    ner = load_NNER()
    print('successfully load model')
    out_file='../CMeEE_test_answer.json'
    ner.predict_to_file(in_file, out_file)