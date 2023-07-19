from typing import Tuple
from enum import Enum

from torch.utils.data import Dataset

import numpy as np

BKG = 0
SIG = 1

class AtlasDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.x = x
        self.y = y
        self.w = w

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.x[idx], self.y[idx], self.w[idx]