import torch

from . import FairseqDataset


class LangIdDataset(FairseqDataset):
    def __init__(self, lang_id) -> None:
        super().__init__()
        self.lang_id = lang_id

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 0

    def collater(self, samples):
        return [self.lang_id] * len(samples)
