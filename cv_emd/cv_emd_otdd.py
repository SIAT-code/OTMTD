import pickle
import torch
import itertools
import numpy as np
from torch.utils.data import Dataset

# from ..wte.utils.distance_otdd import OTDD
from otdd.pytorch.distance import DatasetDistance
from utils import fix_random_seed


class EmbeddingDataset(Dataset):
    def __init__(self, emd_lbl_dir, num_samples=500):
        super(EmbeddingDataset, self).__init__()
        with open(emd_lbl_dir, 'rb') as handle:
            emd_lbl = pickle.load(handle)
        embeddings, labels = emd_lbl['embeddings'], emd_lbl['labels']

        # shuffle and sampling
        shuffle_idxs = list(range(len(labels)))
        fix_random_seed(1145114)
        np.random.shuffle(shuffle_idxs)
        embeddings, labels = embeddings[shuffle_idxs], labels[shuffle_idxs]
        embeddings, labels = embeddings[:num_samples], labels[:num_samples]
        
        self.embeddings, labels = torch.tensor(embeddings), torch.tensor(labels)
        # Define attributes for OTDD
        self.classes = [str(k) for k in labels.unique().numpy()] # list of unique labels string
        self.targets = labels

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.embeddings[idx].reshape(1, -1, 1), self.targets[idx]


if __name__ == '__main__':
    # Build embedding dataset
    num_samples = 10000 # 500
    all_names = ['mnist', 'fashion_mnist', 'cifar10']
    all_datasets = []
    for set_name in all_names:
        all_datasets.append(EmbeddingDataset(
            f"/home/brian/work/OTMTD_GH/cv_emd/emd_lbl/{set_name}_emd_lbl.pkl", num_samples))

    cv_t2t_emd_otdds = {}
    for comb in list(itertools.combinations(range(len(all_datasets)), 2)):
        src_dataset, tgt_dataset = all_datasets[comb[0]], all_datasets[comb[1]]
        # Instantiate distance
        dist = DatasetDistance(src_dataset, tgt_dataset,
                                inner_ot_method = 'exact',
                                debiased_loss = True,
                                feature_cost = 'euclidean',
                                sqrt_method = 'spectral',
                                sqrt_niters=10,
                                precision='single',
                                p = 2, entreg = 1e-1,
                                device='cpu')
        d = dist.distance() # maxsamples=1000
        src_name, tgt_name = all_names[comb[0]], all_names[comb[1]]
        print("OTDD between {} and {} embeddings: {:.2f}".format(src_name, tgt_name,d))
        cv_t2t_emd_otdds[(src_name, tgt_name)] = d.numpy()
    
    with open("/home/brian/work/OTMTD_GH/cv_emd/cv_emd_otdds.pkl", 'wb') as handle:
        pickle.dump(cv_t2t_emd_otdds, handle)
