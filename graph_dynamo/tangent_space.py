import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, issparse
from dynamo.tools.sampling import sample_by_kmeans
from dynamo.tools.utils import nearest_neighbors
from sklearn.linear_model import Lasso

def linear_embedding_velocity(V, PCs):
    return V @ PCs


def local_linear_embedding_velocity(X, V, nbrs_idx, dists=None, d_perc=0.8, m=100, n_pca=10):
    cidx = np.unique(sample_by_kmeans(X, m, return_index=True))
    print(f'{len(cidx)} points are sampled for approximating tangent space.')
    
    Ts = []
    valid_i = []
    for i, idx in enumerate(cidx):
        nbrs_i = nbrs_idx[idx]
        X_ = X[nbrs_i]
        if d_perc:   # pick a ball
            if dists:
                D = dists[nbrs_i]
            else:
                D = cdist(X[idx][None], X_).flatten()
            #print(np.sum(D < np.max(D) * d_perc))
            X_ = X[nbrs_i[D < np.max(D) * d_perc]]

        if X_.shape[0] > n_pca:
            pca = PCA(n_components=n_pca)
            pca.fit(X_)
            Ts.append(pca.components_)
            valid_i.append(i)
    
    cidx = cidx[valid_i]
    
    U = np.zeros(V.shape)
    for i, v in enumerate(V):
        j = nearest_neighbors(X[i], X[cidx], k=1)[0][0]
        T = Ts[j]
        U[i] = np.sum(T.dot(v) * T.T, axis=1)
    
    return U


def tangent_correcting_velocity(X, V, dt=1, k=5):
    Y = X + V * dt

    Y_nbrs = nearest_neighbors(Y, X, k=k)
    Z = np.zeros(Y.shape)
    for i, y_nbrs in enumerate(Y_nbrs):
        Z[i] = np.mean(X[y_nbrs], axis=0)
    
    U = (Z - X) / dt
    return U


def dvf_lasso(X, V, nbrs, alpha=1.0):
    P = np.zeros((X.shape[0], X.shape[0]))
    for i, x in enumerate(X):
        v = V[i]
        idx = nbrs[i]

        # normalized differences
        U = X[idx] - x
        dist = np.linalg.norm(U, axis=1)
        dist[dist == 0] = 1
        U /= dist[:, None]

        clf = Lasso(alpha=alpha)
        clf.fit(U.T, v)
        #print(clf.coef_)
        P[i][idx] = clf.coef_
    return P


def dvf_coopt(X, V, U, nbrs, b=1.0, nonneg=False, norm_diff=False):
    P = np.zeros((X.shape[0], X.shape[0]))

    for i, x in enumerate(X):
        v, u, idx = V[i], U[i], nbrs[i]

        # normalized differences
        D = X[idx] - x
        if norm_diff:
            dist = np.linalg.norm(D, axis=1)
            dist[dist == 0] = 1
            D /= dist[:, None]

        # co-optimization
        u_norm = np.linalg.norm(u)
        def func(w):
            v_ = w @ D

            # cosine similarity between v_ and u
            uv = u_norm * np.linalg.norm(v_)
            if uv > 0:
                sim = u.dot(v_) / uv
            else:
                sim = 0

            # reconstruction error between v_ and v
            loss = v_ - v
            return loss.dot(loss) - b*sim
        
        def fjac(w):
            v_ = w @ D
            v_norm = np.linalg.norm(v_)
            u_ = u / u_norm

            jac_con = 2 * D @ (v_ - v)
            jac_sim = 0 if v_norm == 0 else b / v_norm ** 2 * (v_norm * D @ u_ - v_.dot(u_) * v_ @ D.T / v_norm) 
            return jac_con - jac_sim

        if nonneg:
            bounds = [(0, np.inf)] * D.shape[0]
        else:
            bounds = None

        res = minimize(func, x0=D @ v + 100*np.random.rand(D.shape[0]), jac=fjac, bounds=bounds)
        P[i][idx] = res['x']
    return P


def binary_corr(xi, Xj, vi):
    return np.mean(np.sign(vi) == np.sign(Xj - xi), axis=1)


def cos_corr(xi, Xj, vi):
    D = Xj - xi
    dist = np.linalg.norm(D, axis=1)
    dist[dist == 0] = 1
    D /= dist[:, None]

    v_norm = np.linalg.norm(vi)
    if v_norm == 0:
        v_norm = 1
    
    return D @ vi / v_norm


def corr_kernel(X, V, nbrs, sigma=10, corr_func=binary_corr):
    P = np.zeros((X.shape[0], X.shape[0]))
    for i, x in enumerate(X):
        v, idx = V[i], nbrs[i]

        c = corr_func(x, X[idx], v)
        p = np.exp(c/sigma)
        p /= np.sum(p)
        P[i][idx] = p
    return P


def projection_with_transition_matrix(T, X_emb, correct_density=True, norm_diff=False):
    n = T.shape[0]
    V = np.zeros((n, X_emb.shape[1]))

    if not issparse(T):
        T = csr_matrix(T)

    for i in range(n):
        idx = T[i].indices
        diff_emb = X_emb[idx] - X_emb[i, None]
        if norm_diff:
            diff_emb /= np.linalg.norm(diff_emb, axis=1)[:, None]
        if np.isnan(diff_emb).sum() != 0:
            diff_emb[np.isnan(diff_emb)] = 0
        T_i = T[i].data
        V[i] = T_i.dot(diff_emb)
        if correct_density:
            V[i] -= T_i.mean() * diff_emb.sum(0)

    return V


def density_corrected_transition_matrix(T):
    T = sp.csr_matrix(T, copy=True)

    for i in range(T.shape[0]):
        idx = T[i].indices
        T_i = T[i].data
        T_i -= T_i.mean()
        T[i, idx] = T_i

    return T