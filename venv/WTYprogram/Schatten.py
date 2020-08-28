import time
import math
import scipy
import random
import scipy.io
import itertools
import numpy as np
from imageloader import *
import scipy.sparse as ss
from numpy import linalg as la
import matplotlib.pyplot as plt
from scipy.sparse.linalg import norm
from scipy.linalg import fractional_matrix_power


def OtraceEEC(M, H, full, iter_num, data,experiment):
    print('Schatten-p Norm Minimization begins')
    # parameters
    p = 0.01
    #M m*n input data matrix, Y.*A is the observed elements in the data matrix Y
    #H m*n mask matrix, A(i,j)=1 if Y(i,j) is observed, otherwise A(i,j)=0.(denoted as H in the paper)A
    #p the p value of the Schatten p-Norm
    #st: regularization to avoid singularity of matrix
    #X: recovered data matrix


    # dimension of initial matrix
    m,n = M.shape
    # record intial matrix to calculate rmse
    M_1 = M
    # masked matrix
    M = M * H
    # initialize result
    X = M

    D = 2 / p * fractional_matrix_power((np.dot(X.T, X)), 1 - (p / 2))
    timelist =[]
    resultlist = []
    rmse = []
    obj = []
    start_time = time.perf_counter()
    for iter in range(iter_num):
        print('Schatten-p Norm Minimimization iteration:', iter)
        Lambda = np.zeros(shape = (m,n))
        for i in range(m):
            Lambda[i,:] = 2 * np.dot(M[i,:],np.linalg.pinv(np.dot(np.dot(np.diag(H[i,:]),D),np.diag(H[i,:]))))

        X = 1/2 * np.dot((Lambda * H), D)

        end_time = time.perf_counter()
        timelist.append(end_time - start_time)
        mask_X = X * H
        if data == 'lena' or data =='noisy' or data == 'picture' or data == 'noisy_picture':
            relative_error = la.norm(X - full) / la.norm(full)

            psnr = 10 * math.log10( m * n * (255 ** 2) / la.norm(X - full) ** 2  )
            resultlist.append(psnr)
        if data == 'random':
            relative_error = la.norm(X - full) / la.norm(full)
            resultlist.append(relative_error)

        if data == 'movielens':
            relative_error = la.norm(mask_X  - M_1) / (len(M_1.nonzero()[0])* 4)
            resultlist.append(relative_error)

        diff = M_1 - X
        D = 2/p * fractional_matrix_power(np.dot(X.T, X) ,1 - (p / 2))

    if data == 'lena' or data =='noisy' or data == 'picture' or data == 'noisy_picture':
        image = MatrixToImage(X)
        image.save(r'C:\Users\pc\Desktop\毕设图片\\' + data + '_Schatten_experiment+'+str(experiment)+'.png')

    return timelist, resultlist