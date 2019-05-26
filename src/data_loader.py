from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset
from pytorch_pretrained_bert.tokenization import BertTokenizer


class LabeledDocDataset(Dataset):
    def __init__(self,
                 path: str,
                 tokenizer: BertTokenizer,
                 wlim: Optional[int]):
        self.wlim = wlim
        self.sources, self.targets = self._load(path)
        self.max_phrase_len: int = max(len(phrase) for phrase in self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def __getitem__(self, idx) -> Tuple[List[int], List[int], int]:
        source = self.sources[idx]  # (len)
        mask = [1] * len(source)    # (len)
        target = self.targets[idx]  # ()
        return source, mask, target

    def _load(self, path: str) -> Tuple[List[List[int]], List[int]]:
        sources, targets = [], []
        with open(path) as f:
            for line in f:
                tag, body = line.strip().split('\t')
                assert tag in ['1', '-1']
                targets.append(int(tag == '1'))
                ids: List[int] = []
                for mrph in body.split():
                    if mrph in self.word2id:
                        ids.append(self.word2id[mrph])
                    else:
                        ids.append(self.word2id['<UNK>'])
                if self.wlim is not None and len(ids) > self.wlim:
                    ids = ids[-self.wlim:]  # limit word length
                sources.append(ids)
        assert len(sources) == len(targets)
        return sources, targets
