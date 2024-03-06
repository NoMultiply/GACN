import random

import numpy as np
import torch
import tqdm


def color_print(*args):
    print('\033[1;32m', end='')
    print(*args, end='')
    print('\033[0m')


def progress(data):
    return tqdm.tqdm(data, bar_format='{l_bar}{r_bar}')


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def randint_choice(high, size=None, replace=True, p=None, exclusion=None):
    a = np.arange(high)
    if exclusion is not None:
        if p is None:
            p = np.ones_like(a)
        else:
            p = np.array(p, copy=True)
        p = p.flatten()
        p[exclusion] = 0
    if p is not None:
        p = p / np.sum(p)
    sample = np.random.choice(a, size=size, replace=replace, p=p)
    return sample


class EarlyStopper:
    def __init__(self, max_patience, get_model=None, _print=None, warmup=None, max_score=None, best_score=0):
        self.max_patience = max_patience
        self.best_score = best_score
        self.best_model = None
        self.patience = 0
        self.get_model = get_model
        self._print = _print
        self.warmup = warmup
        self.step = 0
        self.max_score = max_score

    def reset(self):
        self.best_score = 0
        self.best_model = None
        self.patience = 0
        self.step = 0

    def update(self, score):
        self.step += 1
        if score > self.best_score:
            self._print and self._print(f'[!!! Improved !!!] Best score: {self.best_score} -> {score}')
            self.best_score = score
            if self.get_model:
                self.best_model = self.get_model()
            if self.max_score is not None and self.best_score >= self.max_score:
                self._print and self._print(f'[!!! Already Best !!!]')
                return False
            self.patience = 0
        else:
            self._print and self._print(f'[Not Improved] Best score: {self.best_score}, current score: {score}')
            if self.warmup and self.step <= self.warmup:
                self._print and self._print(f'[!!! Warmup Update !!!]')
                if self.get_model:
                    self.best_model = self.get_model()
            else:
                self.patience += 1

            if self.patience >= self.max_patience:
                self._print and self._print(f'[!!! Early Stopping !!!]')
                return False
        return True


def shuffle(*args):
    data = list(zip(*args))
    random.shuffle(data)
    return zip(*data)


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
