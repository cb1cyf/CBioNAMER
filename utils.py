import numpy as np
import unicodedata

def to_array(*args):
    """批量转numpy的array
    """
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results

def is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')

def is_special(ch):
    """判断是不是有特殊含义的符号
    """
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']') 
    #and(ch !='[UNK]')

def stem(token):
    """获取token的“词干”（如果是##开头，则自动去掉##）
    """
    if token[:2] == '##':
        return token[2:]
    else:
        return token

def rematch(tokenizer,text, tokens,):
    """
    给出原始的text和tokenize后的tokens的映射关系
    """
    
    text = text.lower()
    normalized_text, char_mapping = '', []
    for i, ch in enumerate(text):
        ch = ''.join([
            c for c in ch
            if not (ord(c) == 0 or ord(c) == 0xfffd or is_control(c))
        ])
        normalized_text += ch
        char_mapping.extend([i] * len(ch))
    
    i = 0
    j = 0
    text, token_mapping, offset = normalized_text, [], 0
    for token in tokens:
        if is_special(token):
            token_mapping.append([i])
            i = i + 1
        else:
            token = stem(token)
            #print(token)
            start = text[offset:].index(token) + offset
            #print(start)
            end = start + len(token)
            token_mapping.append(char_mapping[start:end])
            offset = end
            j = end - start - 1
            i = i + j + 1

        
    return token_mapping

