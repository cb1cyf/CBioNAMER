# ***model name***

***xxx*** is our method used in [CBLUE](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414) (Chinese Biomedical Language Understanding Evaluation), a benchmark of Nested Named Entity Recognition. We got the ***xx*** price of the benchmark by ***date***.



## Approach

TODO:

picture or paper



## Usage

First, install [PyTorch>=1.7.0](https://pytorch.org/get-started/locally/). There's no restriction on GPU or CUDA.

Then, install this repo as a Python package:

```bash
$ pip install nner
```



## API

The `nner` package provides the following methods:

`nner.load_NNER(model_save_path='./checkpoint/macbert-large_dict.pth', maxlen=512, c_size=9, id2c=_id2c)`

Returns the pretrained model. It will download the model as necessary. The model would use the first CUDA device if there's any, otherwise using CPU instead. 

The `model_save_path` argument specifies the path of the pretrained model weight.

The `maxlen` argument specifies the max length of input sentences. The sentences longer than `maxlen` would be cut off.

The `c_size` argument specifies the number of entity class. Here is `9` for CBLUE.

The `id2c` argument specifies the mapping between id and entity class. By default, the `id2c` argument for CBLUE is:

`_id2c = {0: 'dis', 1: 'sym', 2: 'pro', 3: 'equ', 4: 'dru', 5: 'ite', 6: 'bod', 7: 'dep', 8: 'mic'}`

------

The model returned by `nner.load_NNER()` supports the following methods:

`model.recognize(text: str, threshold=0)`

Given a sentence, returns a list of tuples with recognized entity, the format of the tuple is `[(start_index, end_index, entity_class), ...]`. The `threshold` argument specifies that the returned list only contains the recognized entity with confidence score higher than `threshold`.

`model.predict_to_file(in_file: str, out_file: str)`

Given input and output `.json` file path, the model would do inference according `in_file`, and the recognized entity would be saved in `out_file`. The output file can be submitted to [CBLUE](https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414). The format of input file is like:

```json
[
  {
    "text": "..."
  },
  {
    "text": "..."
  },
  ...
]
```



## Examples

```Python
import nner

NNER = nner.load_NNER()
in_file = './CMeEE_test.json'
out_file = './CMeEE_test_answer.json'
NNER.predict_to_file(in_file, out_file)
```