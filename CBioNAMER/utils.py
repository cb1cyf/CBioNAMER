import numpy as np
import unicodedata

def to_array(*args):
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results

def is_control(ch):
    return unicodedata.category(ch) in ('Cc', 'Cf')

def is_special(ch):
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']') 
    #and(ch !='[UNK]')

def stem(token):
    if token[:2] == '##':
        return token[2:]
    else:
        return token

def rematch(tokenizer,text, tokens,):
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
            start = text[offset:].index(token) + offset
            end = start + len(token)
            token_mapping.append(char_mapping[start:end])
            offset = end
            j = end - start - 1
            i = i + j + 1

        
    return token_mapping

