import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizerFast
from .model import GlobalPointer
from .utils import rematch

#__all__=['model_save_path','model_base','NamedEntityRecognizer','in_file','out_file','predict_to_file']

#########################################

'''
name_tag = 0 # 0代表使用模型名称导入bert 1代表使用模型路径导入bert
model_path = "./pretrain/roberta_wwm_ext_large"
model_save_path = "./checkpoint/roberta_wwm_ext_large"
'''

model_name = "hfl/chinese-macbert-large"
#model_name = "hfl/chinese-roberta-wwm-ext-large"
model_save_path = "./checkpoint/our_model"


model_save_path ="./checkpoint/68.2796_macbert_large/macbert-large"

out_file = './CMeEE_test_answer.json'
in_file = './dataset/normal/CMeEE_test.json'
train_file = './dataset/normal/CMeEE_train.json'
val_file = './dataset/normal/CMeEE_dev.json'

head_size = 64
embedding_size = 1024 # large:1024 or base:768
maxlen = 512 # len of sentence
epochs = 10
batch_size = 8
learning_rate = 5e-6 # 1e-5
tag_fgm = 1 #0代表不使用对抗训练，1代表使用对抗训练
n_eps=0.1

##########################################



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print("Using {} device".format(device))


tokenizer = BertTokenizerFast.from_pretrained(model_name)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
c_size = 9
c2id = {'dis': 0, 'sym': 1, 'pro': 2, 'equ': 3, 'dru': 4, 'ite': 5, 'bod': 6, 'dep': 7, 'mic': 8}
id2c = {0: 'dis', 1: 'sym', 2: 'pro', 3: 'equ', 4: 'dru', 5: 'ite', 6: 'bod', 7: 'dep', 8: 'mic'}

#gpus = [0, 1, 2, 3]
#torch.cuda.set_device('cuda:{}'.format(gpus[0]))




class FGM:
    def __init__(self, model: nn.Module, eps=n_eps):
        self.model = model
        self.eps = eps
        self.backup = {}

    # only attack word embedding
    def attack(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, para in self.model.named_parameters():
            if para.requires_grad and emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]

        self.backup = {}


def load_data(filename):
    """
    加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    for d in json.load(open(filename,encoding='utf-8')):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end, label))
            #categories.add(label)
    return D


def load_test_data(filename):
    """
    加载测试数据，仅有文本
    """
    D = []
    for d in json.load(open(filename,encoding='utf-8')):
        D.append([d['text']])
    return D


train_data = load_data(train_file)
val_data = load_data(val_file)


class GlobalPointerNet(nn.Module):
    def __init__(self, model_name):
        super(GlobalPointerNet, self).__init__()
        self.head = GlobalPointer(c_size, head_size, embedding_size)
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x2 = x1.last_hidden_state
        #last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size))
        # – Sequence of hidden-states at the output of the last layer of the model.

        logits = self.head(x2, mask=attention_mask)
        return logits

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    @staticmethod
    def find_index(offset_mapping, index):
        for idx, internal in enumerate(offset_mapping[1:]):
            if internal[0] <= index < internal[1]:
                return idx + 1
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        label = torch.zeros((c_size, self.maxlen, self.maxlen))
        enc_context = tokenizer(d[0], return_offsets_mapping=True, max_length=self.maxlen, truncation=True,
                                padding='max_length', return_tensors='pt')
        
        enc_context = {key: enc_context[key][0] for key in enc_context.keys() if enc_context[key].shape[0] == 1}
        for entity_info in d[1:]:
            start, end = entity_info[0], entity_info[1]
            offset_mapping = enc_context['offset_mapping']
            start = self.find_index(offset_mapping, start)
            end = self.find_index(offset_mapping, end)
            if start and end and start < self.maxlen and end < self.maxlen:
                label[c2id[entity_info[2]], start, end] = 1
        return enc_context, label

class CustomDataset4test(Dataset):
    def __init__(self, data, tokenizer, maxlen):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    @staticmethod
    def find_index(offset_mapping, index):
        for idx, internal in enumerate(offset_mapping[1:]):
            if internal[0] <= index < internal[1]:
                return idx + 1
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        label = torch.zeros((c_size, self.maxlen, self.maxlen))
        enc_context = tokenizer(d[0], return_offsets_mapping=True, max_length=self.maxlen, truncation=True,
                                padding='max_length', return_tensors='pt')
                                
        enc_context = {key: enc_context[key][0] for key in enc_context.keys() if enc_context[key].shape[0] == 1}

        return enc_context


model = GlobalPointerNet(model_name).to(device)
# model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
training_data = CustomDataset(train_data, tokenizer, maxlen)
valing_data = CustomDataset(val_data, tokenizer, maxlen)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(valing_data, batch_size=batch_size)





def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


def global_pointer_crossentropy(y_true, y_pred):
    """
    给GlobalPointer设计的交叉熵
    """
    bh = y_pred.shape[0] * y_pred.shape[1]
    y_true = torch.reshape(y_true, (bh, -1))
    y_pred = torch.reshape(y_pred, (bh, -1))
    return torch.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    y_pred = torch.greater(y_pred, 0)
    return torch.sum(y_true * y_pred).item(), torch.sum(y_true + y_pred).item()


def train(dataloader, model, loss_fn, optimizer, fgm_flag=0):
    if fgm_flag==1:
        model.train()
        size = len(dataloader.dataset)
        numerate, denominator = 0, 0
        fgm = FGM(model)#fgm
        for batch, (data, y) in enumerate(dataloader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            y = y.to(device)

            pred = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(y, pred)

            temp_n, temp_d = global_pointer_f1_score(y, pred)
            numerate += temp_n
            denominator += temp_d
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            fgm.attack() #fgm 在embedding上添加对抗扰动
            adv_pred = model(input_ids, attention_mask, token_type_ids)#fgm
            loss_adv = loss_fn(y,adv_pred)#fgm 
            loss_adv.backward() #fgm 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore() #fgm 恢复embedding参数
            
            optimizer.step()
            model.zero_grad()
            if batch % 50 == 0:
                loss, current = loss.item(), batch * len(input_ids)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                
        print(f"Train F1: {(2 * numerate / denominator):>4f}")
    else:
        model.train()
        size = len(dataloader.dataset)
        numerate, denominator = 0, 0
        for batch, (data, y) in enumerate(dataloader):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            y = y.to(device)
            pred = model(input_ids, attention_mask, token_type_ids)
            loss = loss_fn(y, pred)
            temp_n, temp_d = global_pointer_f1_score(y, pred)
            numerate += temp_n
            denominator += temp_d
        
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch % 50 == 0:
                loss, current = loss.item(), batch * len(input_ids)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                
        print(f"Train F1: {(2 * numerate / denominator):>4f}")
        


def test(dataloader, loss_fn, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    numerate, denominator = 0, 0
    with torch.no_grad():
        for data, y in dataloader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            y = y.to(device)
            pred = model(input_ids, attention_mask, token_type_ids)
            test_loss += loss_fn(y, pred).item()
            temp_n, temp_d = global_pointer_f1_score(y, pred)
            numerate += temp_n
            denominator += temp_d
    test_loss /= size
    test_f1 = 2 * numerate / denominator
    print(f"Val: F1:{(test_f1):>4f},Avg loss: {test_loss:>8f} \n")
    return test_f1


class NamedEntityRecognizer(object):
    """
    命名实体识别器
    """
    def recognize(self, text, threshold=0):
        
        model_base.eval()
        text = text[0:maxlen]
 
        #tokens = tokenizer.tokenize(text, max_length=maxlen, truncation=True)
        tokens = tokenizer.tokenize(text)
        #print(tokens)
        #text = text[0:maxlen]
        mapping = rematch(tokenizer, text, tokens)
        text = [[str(text)]]
        data = CustomDataset4test(text,tokenizer,maxlen)
        data = DataLoader(data)
        input_ids = []
        attention_mask = []
        token_type_ids = []
        
        for data_1 in data:
            input_ids = data_1['input_ids'].to(device)
            attention_mask = data_1['attention_mask'].to(device)
            token_type_ids = data_1['token_type_ids'].to(device)


        with torch.no_grad():
            scores = model_base(input_ids, attention_mask, token_type_ids)

        entities = []
        lenss = len(mapping)

        
        score = scores[0,:,0:lenss,0:lenss].cpu().numpy()
        score = np.triu(score)

  
        for l, start, end in zip(*np.where(score > threshold)):
            entities.append(
                (mapping[start-1][0], mapping[end-1][-1], id2c[l])
            )
            
        return entities
    



def predict_to_file(in_file, out_file):
    """
    预测到文件
    可以提交到 https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414
    """
    data = json.load(open(in_file))
    for d in data:
        d['entities'] = []
        entities = NER.recognize(d['text'])
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


if __name__ == '__main__':
    
    #以下为训练过程，已注释
    # max_F1 = 0
    # for t in range(epochs):
    #     print(f"Epoch {t + 1}\n-------------------------------")
    #     train(train_dataloader, model, global_pointer_crossentropy, optimizer, tag_fgm)
    #     F1 = test(val_dataloader, global_pointer_crossentropy, model)
    #     if F1 > max_F1:
    #         max_F1 = F1
    #         print(f"Better F1: {(max_F1):>4f}")
    #         torch.save(model,model_save_path)
    #         print('best model saved')
    # print("Done!")
    # print(f"Best F1: {(max_F1):>4f}")
    
    model_base = torch.load(model_save_path)
    NER = NamedEntityRecognizer()
    predict_to_file(in_file, out_file)
    print("answer saved")
