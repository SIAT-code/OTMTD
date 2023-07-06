import random
import torch
import numpy as np


def fix_random_seed(seed):
    torch.manual_seed(seed)                   # Current CPU
    torch.cuda.manual_seed(seed)              # Current GPU
    np.random.seed(seed)                      # Numpy module
    random.seed(seed)                         # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed)          # All GPU (Optional)