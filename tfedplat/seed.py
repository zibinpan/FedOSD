
import torch
import random
import numpy as np


def setup_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 10000)
        print('seed=', seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
