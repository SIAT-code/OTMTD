from .otdd.pytorch.datasets import load_torchvision_data
from .otdd.pytorch.distance import DatasetDistance

# Load data
loaders_src  = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=2000)[0]
loaders_tgt  = load_torchvision_data('USPS',  valid_size=0, resize = 28, maxsize=2000)[0]

# Instantiate distance
dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'], # fold_loaders['train', 'valid', 'test']
                          inner_ot_method = 'exact',
                          debiased_loss = True,
                          feature_cost = 'euclidean',
                          sqrt_method = 'spectral',
                          sqrt_niters=10,
                          precision='single',
                          p = 2, entreg = 1e-1,
                          device='cpu')

d = dist.distance(maxsamples = 1000)
print(f'OTDD(MNIST,USPS)={d:8.2f}')
