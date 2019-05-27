from typing import List, Tuple

from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np


class LabeledDocDataset(Dataset):
    def __init__(self,
                 path: str,
                 max_seq_length: int,
                 tokenizer: BertTokenizer
                 ) -> None:
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.sources, self.targets = self._load(path)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        source: List[int] = self.sources[idx]
        seq_length = len(source)
        if seq_length > self.max_seq_length:
            input_ids = np.array(source[:self.max_seq_length])  # (b, seq)
            input_mask = np.array([1] * self.max_seq_length)    # (b, seq)
        else:
            # Zero-pad up to the sequence length
            pad = [0] * (self.max_seq_length - seq_length)
            input_ids = np.array(source + pad)             # (b, seq)
            input_mask = np.array([1] * seq_length + pad)  # (b, seq)
        target = np.array(self.targets[idx])  # ()
        return input_ids, input_mask, target

    def _load(self, path: str) -> Tuple[List[List[int]], List[int]]:
        sources, targets = [], []
        with open(path) as f:
            for line in f:
                tag, body = line.strip().split(',')  # assuming csv data
                targets.append(int(tag))
                tokens: List[str] = self.tokenizer.tokenize(body)
                sources.append(self.tokenizer.convert_tokens_to_ids(tokens))
        assert len(sources) == len(targets)
        return sources, targets