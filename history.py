import random
import torch
import numpy as np

class History():
    def __init__(self):
        self.s = []

    def clear(self, s):
        del self.s[:]

    def store(self, s):
        self.s.append(s)

    def get_minibatches(self, batch_size, num_batches):
        count = min(batch_size, len(self.s))

        for _ in range(num_batches):
            batch = random.sample(self.s, count)
            yield torch.FloatTensor(batch)
