import numpy as np
import ot
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm import tqdm
import itertools

from .bures_wasserstein import *


class OTDD:
    def __init__(self, device, pt_names=None, ft_names=None, pt_tasks_combs=[], pt_tasks_dict=None,
                gaussian_assumption=True, maxsamples=None):
        self.device = device
        self.assumption = gaussian_assumption
        self.maxsamples = maxsamples
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
        OTDD_distances = np.zeros((len(pt_datasets), len(ft_datasets)))
        for i, pt_dataset in enumerate(pt_datasets):
            pt_X, pt_Y = self.preprocess_dataset(pt_dataset)
            for j, ft_dataset in enumerate(ft_datasets):
                # label-to-label distance
                ft_X, ft_Y = self.preprocess_dataset(ft_dataset)
                print("Computing inter label-to-label distance for {} & {}".format(self.pt_names[i], self.ft_names[j]))
                l2l_distance = self.precompute_dissimilarity(pt_dataset, ft_dataset, symmetric=False)
                # label-to-label indexed samples distance
                l2l_dist_mask = torch.zeros((len(pt_Y), len(ft_Y)), device=l2l_distance.device)
                for y1 in range(l2l_distance.shape[0]):
                    y1_idxs = torch.where(pt_Y==y1)[0].flatten()
                    for y2 in range(l2l_distance.shape[1]):
                        y2_idxs = torch.where(ft_Y==y2)[0].flatten()
                        l2l_dist_mask[np.ix_(y1_idxs.cpu(), y2_idxs.cpu())] = l2l_distance[y1, y2]

                # feature distance
                M = ot.dist(pt_X.numpy(), ft_X.numpy(), metric='euclidean')
                outer_pwdist = ot.emd2(ot.unif(len(pt_X)), ot.unif(len(ft_X)), M, numItermax=100000)

                # OTDD = feature distance + l2l indexed distance
                OTDD_dist = outer_pwdist + l2l_dist_mask.numpy()
                OTDD_dist = np.nanmean(OTDD_dist)
                OTDD_distances[i, j] = OTDD_dist
                print("OTDD distance of {} and {} is: {}".format(self.pt_names[i], self.ft_names[j], OTDD_dist))
        
        return OTDD_distances

    def dissimilarity_for_all_p2(self, pt_datasets, ft_datasets):
        OTDD_distances = np.zeros((len(pt_datasets), len(ft_datasets)))
        for i, pt_dataset in enumerate(pt_datasets):
            pt_X, pt_Y = self.preprocess_dataset(pt_dataset)
            for j, ft_dataset in enumerate(ft_datasets):
                # label-to-label distance
                ft_X, ft_Y = self.preprocess_dataset(ft_dataset)
                print("Computing inter label-to-label distance for {} & {}".format(self.pt_names[i], self.ft_names[j]))
                l2l_distance = self.precompute_dissimilarity(pt_dataset, ft_dataset, symmetric=False)
                # label-to-label indexed samples distance
                l2l_dist_mask = torch.zeros((len(pt_Y), len(ft_Y)), device=l2l_distance.device)
                for y1 in range(l2l_distance.shape[0]):
                    y1_idxs = torch.where(pt_Y==y1)[0].flatten()
                    for y2 in range(l2l_distance.shape[1]):
                        y2_idxs = torch.where(ft_Y==y2)[0].flatten()
                        l2l_dist_mask[np.ix_(y1_idxs.cpu(), y2_idxs.cpu())] = l2l_distance[y1, y2]

                # feature distance
                M = ot.dist(pt_X.numpy(), ft_X.numpy(), metric='euclidean')
                M = pow(M, 2)
                outer_pwdist = ot.emd2(ot.unif(len(pt_X)), ot.unif(len(ft_X)), M, numItermax=100000)
                outer_pwdist = np.sqrt(outer_pwdist)

                # OTDD = feature distance + l2l indexed distance
                OTDD_dist = outer_pwdist + l2l_dist_mask.numpy()
                OTDD_dist = np.nanmean(OTDD_dist)
                OTDD_distances[i, j] = OTDD_dist
                print("OTDD distance of {} and {} is: {}".format(self.pt_names[i], self.ft_names[j], OTDD_dist))
        
        return OTDD_distances

    def t2t_distance(self, pt_datasets, ft_datasets, p):
        OTDD_distances = np.zeros((len(pt_datasets), len(ft_datasets)))
        for i, pt_dataset in enumerate(pt_datasets):
            pt_X, pt_Y = self.preprocess_dataset(pt_dataset)
            for j, ft_dataset in enumerate(ft_datasets):
                # label-to-label distance
                ft_X, ft_Y = self.preprocess_dataset(ft_dataset)
                print("Computing inter label-to-label distance for {} & {}".format(self.pt_names[i], self.ft_names[j]))
                l2l_distance = self.precompute_dissimilarity(pt_dataset, ft_dataset, symmetric=False)
                # label-to-label indexed samples distance
                l2l_dist_mask = torch.zeros((len(pt_Y), len(ft_Y)), device=l2l_distance.device)
                # print(l2l_distance)
                for y1 in range(l2l_distance.shape[0]):
                    y1_idxs = torch.where(pt_Y==y1)[0].flatten()
                    for y2 in range(l2l_distance.shape[1]):
                        y2_idxs = torch.where(ft_Y==y2)[0].flatten()
                        l2l_dist_mask[np.ix_(y1_idxs.cpu(), y2_idxs.cpu())] = l2l_distance[y1, y2]
                l2l_dist_mask = pow(l2l_dist_mask, p)

                # feature distance
                Dx = ot.dist(pt_X.numpy(), ft_X.numpy(), metric='euclidean')
                Dx = pow(Dx, p)

                Dz = (Dx + l2l_dist_mask.numpy()) / p
                # print(np.any(np.isnan(Dz)))
                outer_pwdist = ot.emd2(ot.unif(len(pt_X)), ot.unif(len(ft_X)), Dz, numItermax=100000)
                OTDD_dist = outer_pwdist
                OTDD_distances[i, j] = OTDD_dist
                print("OTDD distance of {} and {} is: {:.2f}".format(self.pt_names[i], self.ft_names[j], OTDD_dist))
        
        return OTDD_distances

    def baseline1_sep_mean(self, pt_datasets, ft_datasets, p):
        OTDD_distances = self.t2t_distance(pt_datasets, ft_datasets, p)
        self.OTDD_distances = OTDD_distances
        all_combs_mean_dist = np.zeros((len(self.pt_tasks_combs), len(self.ft_names)))
        for i, pt_tasks_comb in enumerate(self.pt_tasks_combs):
            comb_mean_dist = OTDD_distances[np.ix_([self.pt_tasks_dict[t] for t in pt_tasks_comb], range(OTDD_distances.shape[1]))].mean(axis=0)
            all_combs_mean_dist[i] = comb_mean_dist

        return all_combs_mean_dist

    def baseline2_merge_mean(self, pt_datasets, ft_datasets, p):
        def l2l_dist_to_mask(l2l_distance, pt_Y, ft_Y):
            """ Indexed cost mask with l2l distance. """
            l2l_dist_mask = np.zeros((len(pt_Y), len(ft_Y)))
            for y1 in range(l2l_distance.shape[0]):
                y1_idxs = np.where(pt_Y==y1)[0].flatten()
                for y2 in range(l2l_distance.shape[1]):
                    y2_idxs = np.where(ft_Y==y2)[0].flatten()
                    l2l_dist_mask[np.ix_(y1_idxs, y2_idxs)] = l2l_distance[y1, y2]
            return l2l_dist_mask

        print("Precomputing distances of all (PT)label-to-(FT)label combinations...")
        all_l2l_dist_masks = np.zeros((len(pt_datasets), len(ft_datasets)), dtype=object)
        for i, pt_dataset in enumerate(pt_datasets):
            pt_X, pt_Y = self.preprocess_dataset(pt_dataset)
            for j, ft_dataset in enumerate(ft_datasets):
                # label-to-label distance
                _, ft_Y = self.preprocess_dataset(ft_dataset)
                print("Computing inter label-to-label distance for {} & {}".format(self.pt_names[i], self.ft_names[j]))
                l2l_distance = self.precompute_dissimilarity(pt_dataset, ft_dataset, symmetric=False)
                l2l_dist_mask = l2l_dist_to_mask(l2l_distance.numpy(), pt_Y, ft_Y)
                all_l2l_dist_masks[i, j] = l2l_dist_mask

        print("Computing OTDD...")
        all_combs_mean_dist = np.zeros((len(self.pt_tasks_combs), len(self.ft_names)))
        for i, pt_tasks_comb in enumerate(self.pt_tasks_combs):
            for j, ft_dataset in enumerate(ft_datasets):
                ft_X, _ = self.preprocess_dataset(ft_dataset)
                # comb_mean_l2l_mask = all_l2l_dist_masks[np.ix_([self.pt_tasks_dict[t] for t in pt_tasks_comb], [j])]
                comb_mean_l2l_mask = [all_l2l_dist_masks[self.pt_tasks_dict[t], j] for t in pt_tasks_comb]
                comb_mean_l2l_mask = np.mean(comb_mean_l2l_mask, axis=0).astype(np.float32)
                Dy = pow(comb_mean_l2l_mask, p)

                # feature distance
                Dx = ot.dist(pt_X.numpy(), ft_X.numpy(), metric='euclidean') # all pretrainings share the same features
                Dx = pow(Dx, p)

                Dz = (Dx + Dy) / p
                outer_pwdist = ot.emd2(ot.unif(len(pt_X)), ot.unif(len(ft_X)), Dz, numItermax=100000)
                OTDD_dist = outer_pwdist
                all_combs_mean_dist[i, j] = OTDD_dist
                print("OTDD distance of {} and {} is: {:.2f}".format(self.pt_names[i], self.ft_names[j], OTDD_dist))
        
        return all_combs_mean_dist

    def combination_otdd(self, pt_datasets, ft_datasets, p):
        """ 按照任务-任务组合方式用OTDD计算, 为了让OTMTD能公平比较。仅用于计算运行时间
        """
        all_datasets = pt_datasets + ft_datasets
        all_names = self.pt_names + self.ft_names
        dataset_combinations = list(itertools.combinations(range(len(all_datasets)), 2))
        for t1_idx, t2_idx in dataset_combinations:
            pt_dataset = all_datasets[t1_idx]
            ft_dataset = all_datasets[t2_idx]
            pt_X, pt_Y = self.preprocess_dataset(pt_dataset)
            ft_X, ft_Y = self.preprocess_dataset(ft_dataset)
            print("Computing inter label-to-label distance for {} & {}".format(all_names[t1_idx], all_names[t2_idx]))
            l2l_distance = self.precompute_dissimilarity(pt_dataset, ft_dataset, symmetric=False)
            # label-to-label indexed samples distance
            l2l_dist_mask = torch.zeros((len(pt_Y), len(ft_Y)), device=l2l_distance.device)
            # print(l2l_distance)
            for y1 in range(l2l_distance.shape[0]):
                y1_idxs = torch.where(pt_Y==y1)[0].flatten()
                for y2 in range(l2l_distance.shape[1]):
                    y2_idxs = torch.where(ft_Y==y2)[0].flatten()
                    l2l_dist_mask[np.ix_(y1_idxs.cpu(), y2_idxs.cpu())] = l2l_distance[y1, y2]
            l2l_dist_mask = pow(l2l_dist_mask, p)

            # feature distance
            Dx = ot.dist(pt_X.numpy(), ft_X.numpy(), metric='euclidean')
            Dx = pow(Dx, p)

            Dz = (Dx + l2l_dist_mask.numpy()) / p
            # print(np.any(np.isnan(Dz)))
            outer_pwdist = ot.emd2(ot.unif(len(pt_X)), ot.unif(len(ft_X)), Dz, numItermax=100000)
            OTDD_dist = outer_pwdist
            print("OTDD distance of {} and {} is: {:.2f}".format(all_names[t1_idx], all_names[t2_idx], OTDD_dist))