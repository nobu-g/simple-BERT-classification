# Document classificaton on BERT

## Set up

1. clone this repository
2. `pipenv sync --dev`
3. `git clone git@github.com:huggingface/pytorch-pretrained-BERT.git`
4. comment out `text = self._tokenize_chinese_chars(text)` in tokenization.py
5. `pip install ./pytorch-pretrained-BERT`
6. `pipenv shell`

## Train
`python src/train.py --bert-model <path/to/BERT/model> --train <path/to/train/file> --vlalid <path/to/valid/file> -d <gpu_id>`

## Test
`python src/test.py --bert-model <path/to/BERT/model> --test <path/to/test/file> -d <gpu_id>`

## Dataset
Dataset is assumed to be CSV format like
`<label>,<documnt>`.

`<document>` needs to be splitted into morphemes.
