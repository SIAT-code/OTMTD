import numpy as np
import ot
import geomloss
import torch
import math


class OTNCE:
    def __init__(self, backend='numpy', distMetric='euclidean', numItermax=1e5, return_OT=False):
        assert backend in ['numpy', 'torch'], "Only support numpy or torch backend!"
        self.backend = backend
        self.distMetric = distMetric
        self.numItermax = numItermax
        self.return_OT = return_OT
    
    def otnce(self, X_src, Y_src, X_tgt, Y_tgt):
        """ Compute OT-based Negative Conditional Entropy,
            return NCE and the corresponding coupling matrix P and Wasserstein distance W.
        """
        P, W = self.solve_OT(X_src, X_tgt)
        NCE = self.compute_NCE(P, Y_src, Y_tgt)
        return (NCE, P, W) if self.return_OT else NCE

    def solve_OT(self, X_src, X_tgt):
        """ Solve OT problem, compute optimal coupling matrix P
            and p Wesserstein distance W (with squared euclidean distance).
        """
        if self.backend == 'torch':
            cost_function = lambda x, y: geomloss.utils.squared_distances(x, y)
            C = cost_function(X_src, X_tgt).numpy()
        elif self.backend == 'numpy':
            C = pow(ot.dist(X_src, X_tgt, metric=self.distMetric), 2)
            # C = ot.dist(X_src, X_tgt, metric=self.distMetric)
        
        # Coupling matrix computation of geomloss is complecated, use ot uniformly
        P = ot.emd(ot.unif(X_src.shape[0]), ot.unif(X_tgt.shape[0]), C, numItermax=self.numItermax)
        W = pow((P * C), 1/2).sum() # (P * C).sum()

        return P, W

    def compute_NCE(self, P, Y_src, Y_tgt):
        """ Compute Conditional Entropy for source and target labels based on coupling matrix.
        """
        # Joint distribution of source and target label: P(ys, yt)
        src_uniq_labels = np.unique(Y_src)
        tgt_uniq_labels = np.unique(Y_tgt)
        P_src_tgt = np.zeros((len(src_uniq_labels), len(tgt_uniq_labels)))
        for y1 in src_uniq_labels:
            y1_idxs = np.where(Y_src==y1)[0].flatten()
            for y2 in tgt_uniq_labels:
                y2_idxs = np.where(Y_tgt==y2)[0].flatten()
                # Slice all matching rows and columns of coupling matrix P, then sum
                P_src_tgt[y1, y2] = P[np.ix_(y1_idxs, y2_idxs)].sum() # np.sum(P[Rows, Columns])
        
        # Marginal distribution of source label: P(ys)
        P_src = P_src_tgt.sum(axis=1)

        # Conditional Entropy: H(Yt|Ys)=H(Ys, Yt)-H(Ys)=-∑yt∊Yt ∑ys∊Ys P(ys, yt)log(P(ys, yt)/P(ys))
        ce = 0.0
        for y1 in src_uniq_labels:
            P_y1 = P_src[y1]
            for y2 in tgt_uniq_labels:
                if P_src_tgt[y1, y2] != 0:
                    ce += -(P_src_tgt[y1, y2] * math.log(P_src_tgt[y1, y2] / P_y1))
        return -ce # negative ce


class JCOTCE_(OTNCE):
    def __init__(self):
        super(JCOTCE, self).__init__()
        pass
    
    def jcotce(self, otdd_coupling_matrix, Y_src, Y_tgt):
        jcotce = self.compute_NCE(otdd_coupling_matrix, Y_src, Y_tgt)
        return jcotce

class JCOTCE:
    def __init__(self):
        pass
    
    def jcotce(self, otdd_coupling_matrix, Y_src, Y_tgt):
        jcotce = self.compute_NCE(otdd_coupling_matrix, Y_src, Y_tgt)
        return jcotce

    def compute_NCE(self, P, Y_src, Y_tgt):
        """ Compute Conditional Entropy for source and target labels based on coupling matrix.
        """
        # Joint distribution of source and target label: P(ys, yt)
        src_uniq_labels = np.unique(Y_src)
        tgt_uniq_labels = np.unique(Y_tgt)
        P_src_tgt = np.zeros((len(src_uniq_labels), len(tgt_uniq_labels)))
        for y1 in src_uniq_labels:
            y1_idxs = np.where(Y_src==y1)[0].flatten()
            for y2 in tgt_uniq_labels:
                y2_idxs = np.where(Y_tgt==y2)[0].flatten()
                # Slice all matching rows and columns of coupling matrix P, then sum
                P_src_tgt[y1, y2] = P[np.ix_(y1_idxs, y2_idxs)].sum() # np.sum(P[Rows, Columns])
        
        # Marginal distribution of source label: P(ys)
        P_src = P_src_tgt.sum(axis=1)

        # Conditional Entropy: H(Yt|Ys)=H(Ys, Yt)-H(Ys)=-∑yt∊Yt ∑ys∊Ys P(ys, yt)log(P(ys, yt)/P(ys))
        ce = 0.0
        for y1 in src_uniq_labels:
            P_y1 = P_src[y1]
            for y2 in tgt_uniq_labels:
                if P_src_tgt[y1, y2] != 0:
                    ce += -(P_src_tgt[y1, y2] * math.log(P_src_tgt[y1, y2] / P_y1))
        return -ce # negative ce