# from .poolings import SWE
# from .torchswd import SlicedWD
import time, os

import numpy as np
import ot
import torch
import torch.nn as nn
from sklearn.manifold import MDS
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm

from .bures_wasserstein import *


class label_mds:
    def __init__(self, emb_dim, device, pt_class_nums=None, ft_class_nums=None, pt_names=None, ft_names=None,
                pt_tasks_combs=[], pt_tasks_dict=None, gaussian_assumption=True, maxsamples=None):
        self.dim = emb_dim
        self.device = device
        self.assumption = gaussian_assumption
        self.maxsamples = maxsamples
        self.pt_class_nums = pt_class_nums
        self.ft_class_nums = ft_class_nums
        self.pt_names = pt_names
        self.ft_names = ft_names
        self.pt_tasks_combs = pt_tasks_combs
        self.pt_tasks_dict = pt_tasks_dict

    def preprocess_dataset(self, X):
        if isinstance(X, DataLoader):
            loader = X
        elif isinstance(X, Dataset):
            if self.maxsamples and len(X) > self.maxsamples:
                idxs = np.sort(np.random.choice(
                    len(X), self.maxsamples, replace=False))
                sampler = SubsetRandomSampler(idxs)
                loader = DataLoader(X, sampler=sampler, batch_size=64)
            else:
                # No subsampling
                loader = DataLoader(X, batch_size=64)

        X = []
        Y = []

        # for batch in tqdm(loader, leave=False):
        for batch in loader:
            x = batch[0]
            y = batch[1]
            X.append(x.squeeze().view(x.shape[0], -1))
            Y.append(y.squeeze())

        X = torch.cat(X).to(self.device)
        Y = torch.cat(Y).to(self.device)

        return X, Y

    def precompute_dissimilarity(self, X1, X2=None, symmetric=True, src_class_num=None, tgt_class_num=None):
        X1, Y1 = self.preprocess_dataset(X1)
        if X2 is not None:
            X2, Y2 = self.preprocess_dataset(X2)
        else:
            Y2 = None
            X2 = None
            M2 = None
            S2 = None
        if self.assumption:
            M1, S1 = self.get_gaussian_stats(X1, Y1)
            if X2 is not None:
                M2, S2 = self.get_gaussian_stats(X2, Y2)

            D = efficient_pwdist_gauss(M1, S1, M2, S2, sqrtS1=None, sqrtS2=None,
                                       symmetric=symmetric, diagonal_cov=False, commute=False,
                                       sqrt_method='ns', sqrt_niters=20, sqrt_pref=0,
                                       device=self.device, nworkers=1,
                                       return_dmeans=False, return_sqrts=False)
        else:
            def distance(Xa, Xb):
                C = ot.dist(Xa, Xb, metric='euclidean').cpu().numpy()
                return torch.tensor(ot.emd2(ot.unif(Xa.shape[0]), ot.unif(Xb.shape[0]), C))
            D = torch.zeros((src_class_num, tgt_class_num),
                            device=self.device)
            if symmetric:
                for i in range(src_class_num):
                    for j in range(i+1, tgt_class_num):
                        D[i, j] = distance(X1[Y1 == i], X1[Y1 == j]).item()
                        D[j, i] = D[i, j]
            else:
                for i in range(src_class_num):
                    for j in range(tgt_class_num):
                        D[i, j] = distance(X1[Y1 == i], X2[Y2 == j]).item()
        return D

    def get_gaussian_stats(self, X, Y):
        labels, _ = torch.sort(torch.unique(Y))
        means = torch.stack([torch.mean(X[Y == y].float(), dim=0)
                            for y in labels], dim=0)
        cov = torch.stack([torch.cov(X[Y == y].T) for y in labels], dim=0)
        return means, cov

    def dissimilarity_for_all(self, pt_datasets, ft_datasets):
        lpt, lft = len(pt_datasets), len(ft_datasets)
        l = lpt + lft
        distance_array = np.full((self.pt_class_nums.sum()+self.ft_class_nums.sum(), self.pt_class_nums.sum()+self.ft_class_nums.sum()), 0)

        # upper right
        current_row_coord = 0
        for i in range(l-1):
            src_dataset = pt_datasets[i] if i<lpt else ft_datasets[i-lpt]
            src_class_num = self.pt_class_nums[i] if i<lpt else self.ft_class_nums[i-lpt]
            src_name = self.pt_names[i] if i<lpt else self.ft_names[i-lpt]
            # current_col_coord = 0
            current_col_coord = current_row_coord + src_class_num
            for j in range(i+1, l):
                tgt_dataset = pt_datasets[j] if j<lpt else ft_datasets[j-lpt]
                tgt_class_num = self.pt_class_nums[j] if j<lpt else self.ft_class_nums[j-lpt]
                tgt_name = self.pt_names[j] if j<lpt else self.ft_names[j-lpt]
                print("Computing inter label-to-label distance for {} & {}".format(src_name, tgt_name))
                distance_array[current_row_coord: current_row_coord+src_class_num, current_col_coord: current_col_coord+tgt_class_num] = \
                    np.asarray(self.precompute_dissimilarity(
                        src_dataset, tgt_dataset, symmetric=False).cpu())
                current_col_coord += tgt_class_num
            current_row_coord += src_class_num

        # upper right + lower left
        distance_array = distance_array + distance_array.T - np.diag(np.diag(distance_array)) # symmetric padding
        
        # diagnal line
        current_coord = 0
        for i in range(l):
            dataset = pt_datasets[i] if i<lpt else ft_datasets[i-lpt]
            class_num = self.pt_class_nums[i] if i<lpt else self.ft_class_nums[i-lpt]
            task_name = self.pt_names[i] if i<lpt else self.ft_names[i-lpt]
            print("Computing intra label-to-label distance for {}".format(task_name))
            distance_array[(current_coord): (current_coord+class_num), (current_coord): (current_coord+class_num)] = \
                np.asarray(self.precompute_dissimilarity(dataset).cpu())
            current_coord += class_num
        self.dm = distance_array
        return distance_array # symmetric (pt_class_nums.sum()+ft_class_nums.sum(), pt_class_nums.sum()+ft_class_nums.sum()

    def embedding(self, pt_datasets, ft_datasets, return_t2t=False):
        lpt, lft = len(pt_datasets), len(ft_datasets)
        distance_array = self.dissimilarity_for_all(pt_datasets, ft_datasets) # (pt_class_nums.sum()+ft_class_nums.sum(), pt_class_nums.sum()+ft_class_nums.sum()
        # 'precomputed' requires symetric input, dissim[i, j] should be the dissimilarity between the ith and the jth input points
        embedding = MDS(n_components=self.dim, n_init=10, max_iter=10000, dissimilarity='precomputed') # , n_jobs=os.cpu_count()//4
        mds = embedding.fit_transform(distance_array) # (pt_class_nums.sum()+ft_class_nums.sum(), lbl_emb_dim)

        # pre-training tasks label embedding
        ptdata, pt_labels = None, []
        current_coord = 0
        for idx, dataset in enumerate(pt_datasets):
            class_num = self.pt_class_nums[idx]
            X, Y = self.preprocess_dataset(dataset)
            label_emb = mds[current_coord: current_coord+class_num] # (class_num, lbl_emb_dim)
            labels = torch.stack([torch.from_numpy(label_emb[target]) for target in Y], dim=0).squeeze(1).to(self.device) # (num_samples, lbl_emb_dim)
            if idx == 0:
                ptdata = X.to(self.device) # (num_samples, feat_dim)
            pt_labels.append(labels) # (lpt, num_samples, lbl_emb_dim) 
            current_coord += class_num
        # pt tasks combinations
        comb_ptdata_label_pairs = []
        for comb in self.pt_tasks_combs:
            comb_labels = [pt_labels[self.pt_tasks_dict[task]] for task in comb]
            mean_labels = torch.zeros((len(ptdata), self.dim), device=self.device)
            for lb in comb_labels:
                mean_labels += lb
            mean_labels /= len(comb_labels)
            Z = torch.cat((ptdata, mean_labels), dim=1) # (num_samples, feat_dim + lbl_emb_dim)
            comb_ptdata_label_pairs.append(Z)  # (lft, num_samples, feat_dim + lbl_emb_dim)

        # fine-tuning tasks label embedding
        ftdata_label_pairs = []
        for idx, dataset in enumerate(ft_datasets):
            class_num = self.ft_class_nums[idx]
            X, Y = self.preprocess_dataset(dataset)
            label_emb = mds[current_coord: current_coord+class_num] # (class_num, lbl_emb_dim)
            labels = torch.stack([torch.from_numpy(label_emb[target])
                                 for target in Y], dim=0).squeeze(1).to(self.device) # (num_samples, lbl_emb_dim)
            Z = torch.cat((X.to(self.device), labels), dim=1) # (num_samples, feat_dim + lbl_emb_dim)
            ftdata_label_pairs.append(Z)  # (lft, num_samples, feat_dim + lbl_emb_dim)
            current_coord += class_num

        if not return_t2t:
            return mds, comb_ptdata_label_pairs, ftdata_label_pairs
        else:
            sole_ptdata_label_pairs = []
            for pt_label in pt_labels:
                Z = torch.cat((ptdata, pt_label), dim=1)
                sole_ptdata_label_pairs.append(Z)
            return mds, comb_ptdata_label_pairs, ftdata_label_pairs, sole_ptdata_label_pairs
            

class WTE:
    def __init__(self, label_dim, device, pt_class_nums=None, ft_class_nums=None, pt_names=None, ft_names=None,
                pt_tasks_combs=[], pt_tasks_dict=None, gaussian_assumption=True, maxsamples=None):
        if pt_tasks_dict is None:
            pt_tasks_dict = {'mlm': 0, 'domain': 1, 'region': 2, 'motif': 3, 'go': 4}
        # instantiate label embedder
        self.label_embedder = label_mds(label_dim, device, pt_class_nums, ft_class_nums, pt_names, ft_names,
                                        pt_tasks_combs, pt_tasks_dict, gaussian_assumption, maxsamples)

    def _wte(self, X, ref):
        ref_size = ref.shape[0]
        X = X.float() # (num_samples, feat_dim + lbl_emb_dim)
        C = ot.dist(X.cpu(), ref).cpu().numpy() # (num_samples, ref_size)
        # Calculating the transport plan
        gamma = torch.from_numpy(ot.emd(ot.unif(X.shape[0]), ot.unif(ref_size), C, numItermax=100000)).float() # (num_samples, ref_size) # , numThreads=8
        # Calculating the transport map via barycenter projection
        # (ref_size, num_samples) x (num_samples, feat_dim + lbl_emb_dim) -> (ref_size, feat_dim + lbl_emb_dim)
        f = (torch.matmul((ref_size*gamma).T, X.cpu())-ref) / np.sqrt(ref_size)
        return f

    # def _wte(self, X, ref):
    #     ref_size = ref.shape[0]
    #     X = X.float() # (num_samples, feat_dim + lbl_emb_dim)
    #     C = ot.dist(ref, X.cpu()).cpu().numpy() # (num_samples, ref_size)
    #     # Calculating the transport plan
    #     gamma = torch.from_numpy(ot.unif(ref_size), ot.emd(ot.unif(X.shape[0]), C, numItermax=100000)).float() # (num_samples, ref_size) # , numThreads=8
    #     # Calculating the transport map via barycenter projection
    #     # (ref_size, num_samples) x (num_samples, feat_dim + lbl_emb_dim) -> (ref_size, feat_dim + lbl_emb_dim)
    #     f = (ref_size*torch.matmul(gamma, X.cpu())-ref) / np.sqrt(ref_size)
    #     return f

    def cwte(self, pt_datasets, ft_datasets, ref, return_t2t=False):
        print('Embedding labels...')
        t0 = time.time()
        if not return_t2t:
            _, comb_ptdata_label_pairs, ftdata_label_pairs = self.label_embedder.embedding(pt_datasets, ft_datasets)
        else:
            _, comb_ptdata_label_pairs, ftdata_label_pairs, sole_ptdata_label_pairs = \
                                            self.label_embedder.embedding(pt_datasets, ft_datasets, return_t2t=True)
        print("Finish label embedding in {:.1f} mins".format((time.time()-t0)/60.))
        
        print('Wasserstein embedding...')

        # pre-training tasks combinations
        pt_task_comb_embs = [self._wte(X, ref) for X in comb_ptdata_label_pairs]
        pt_task_comb_embs = torch.stack(pt_task_comb_embs, dim=0).squeeze(0) # (dataset_num, ref_size, feat_dim + lbl_emb_dim)
        # downstream (fine-tuning) tasks
        ft_task_embs = [self._wte(X, ref) for X in ftdata_label_pairs]
        ft_task_embs = torch.stack(ft_task_embs, dim=0).squeeze(0) # (dataset_num, ref_size, feat_dim + lbl_emb_dim)
        print("Finish WTE in {:.1f} mins".format((time.time()-t0)/60.))
        if not return_t2t:
            return pt_task_comb_embs, ft_task_embs
        else:
            pt_task_sole_embs = [self._wte(X, ref) for X in sole_ptdata_label_pairs]
            pt_task_sole_embs = torch.stack(pt_task_sole_embs, dim=0).squeeze(0)
            return pt_task_comb_embs, ft_task_embs, pt_task_sole_embs

        
        
