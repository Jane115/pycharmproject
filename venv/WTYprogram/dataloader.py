import os
import numpy as np
import scipy.sparse as ss

def dataloader_movielens (matrixsize):
    fname = 'dataset/ml-1m/ratings.dat'
    max_uid = 0
    max_vid = 0
    users = []
    movies = []
    ratings = []
    first_line_flag = True
    with open(fname) as f:
        for line in f:
            tks = line.strip().split('::')
            # tks = m
            if first_line_flag:
                max_uid = int(tks[0])
                max_vid = int(tks[1])
                first_line_flag = False
                continue
            max_uid = max(max_uid, int(tks[0]))
            max_vid = max(max_vid, int(tks[1]))
            users.append((int(tks[0]) - 1))
            movies.append(int(tks[1]) - 1)
            ratings.append(int(tks[2]))
    #(row, column), rating
    # original sparse matrix
    M_sparse = ss.csr_matrix((ratings, (users, movies)), shape=(max_uid, max_vid))
    masked_matrix_ndarray = M_sparse.A[:matrixsize, :matrixsize]
    mask = np.int64(masked_matrix_ndarray > 0)
    return masked_matrix_ndarray, mask, masked_matrix_ndarray

def dataloader_random(matrixsize, sp, rank):
    # create low-rank random matrix. dot product of two matrices
    # random m x k matrix
    L = np.random.randn(matrixsize, rank)
    # random k x n matrix
    R = np.random.randn(rank, matrixsize)
    # first row multiply first column. dot product. return m x n matrix
    M_ndarray = np.dot(L, R)
    # create mask matrix
    mask = np.random.choice([0, 1], (matrixsize, matrixsize), p=[(1-sp), sp])
    masked_ndarray = mask * M_ndarray

    return masked_ndarray,mask,M_ndarray

def dataloader_noisy(matrixsize, sp, rank):
    # create low-rank random matrix. dot product of two matrices
    # random m x k matrix
    L = np.random.randn(matrixsize, rank)
    # random k x n matrix
    R = np.random.randn(rank, matrixsize)
    # first row multiply first column. dot product. return m x n matrix
    M_ndarray = np.dot(L, R)
    # create mask matrix
    mask = np.random.choice([0, 1], (matrixsize, matrixsize), p=[(1-sp), sp])
    masked_ndarray = mask * M_ndarray

    return masked_ndarray,mask,M_ndarray