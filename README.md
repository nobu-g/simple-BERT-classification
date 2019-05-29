# Document classificaton on BERT

This is a minimal PyTorch sample for document classification task on BERT.

With this script, you can fine-tune BERT and test it.

And if you run `bert_sample.ipynb` on jupyter notebook, you can learn basic I/O of BertModel.


## Set up

1. clone this repository
2. `pipenv sync --dev`
3. `git clone git@github.com:huggingface/pytorch-pretrained-BERT.git`
4. comment out `text = self._tokenize_chinese_chars(text)` in tokenization.py
5. `pip install ./pytorch-pretrained-BERT`
6. `pipenv shell`


## Train

```
$ python src/train.py --bert-model <path/to/BERT/model> --train <path/to/train/file> --vlalid <path/to/valid/file> --num-labels <number of document classes> --save-path result/model.bin -d <gpu_id>
```

example:
```
$ python src/train.py -b 16 -e 5 --bert-model /larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE --train-file /mnt/hinoki/ueda/shinjin2018/jumanpp.midasi/train.csv --valid-file /mnt/hinoki/ueda/shinjin2018/jumanpp.midasi/valid.csv --max-seq-length 128 --save-path result/model.bin -d 0
```


## Test

```
$ python src/test.py --bert-model <path/to/BERT/model> --test <path/to/test/file> --num-labels <number of document classes> --load-path result/model.bin -d <gpu_id>
```

example:
```
$ python src/test.py -b 32 --bert-model /larch/share/bert/Japanese_models/Wikipedia/L-12_H-768_A-12_E-30_BPE --test-file /mnt/hinoki/ueda/shinjin2018/jumanpp.midasi/test.csv --max-seq-length 128 --load-path result/model.bin -d 0
```


## Dataset
Dataset is assumed to be CSV format like
`<label>,<documnt>`.

`<document>` needs to be splitted into morphemes.


## Reference
[BERT [Devlin+, 2018]](https://arxiv.org/abs/1810.04805)
