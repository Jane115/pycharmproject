import math
import time
import random
import scipy.io
import numpy as np
from imageloader import *
import scipy.sparse as ss
from numpy import linalg as la
from sparsesvd import sparsesvd
from scipy.sparse.linalg import norm


def SVT(M1, full, iter_num, data,experiment):
    print('SVT begins')
    n1, n2 = M1.shape
    total_num = len(M1.nonzero()[0])
    proportion = 1.0
    idx = random.sample(range(total_num), int(total_num * proportion))
    Omega = (M1.nonzero()[0][idx], M1.nonzero()[1][idx])
    tau = 20000
    delta = 2
    maxiter = iter_num
    tol = 0.001
    incre = 5

    # SVT
    r = 0
    b = M1[Omega]
    P_Omega_M = ss.csr_matrix((np.ravel(b), Omega), shape=(n1, n2))
    normProjM = norm(P_Omega_M)
    k0 = np.ceil(tau / (delta * normProjM))
    Y = k0 * delta * P_Omega_M
    iternum = 0
    rmse = []
    timelist = []
    resultlist = []

    start_time = time.perf_counter()
    for k in range(maxiter):
        print('SVT iteration:', k)
        s = r + 1

        while True:
            u1, s1, v1 = sparsesvd(ss.csc_matrix(Y), s)
            if s1[s - 1] <= tau: break
            s = min(s + incre, n1, n2)
            if s == min(n1, n2): break

        r = np.sum(s1 > tau)
        U = u1.T[:, :r]
        V = v1[:r, :]
        S = s1[:r] - tau
        x = (U * S).dot(V)
        x_omega = ss.csr_matrix((x[Omega], Omega), shape=(n1, n2))
        x_omega_ndarray = x_omega.A
        end_time = time.perf_counter()
        timelist.append(end_time-start_time)
        if data == 'lena' or data == 'noisy' or data == 'picture' or data == 'noisy_picture':
            relative_error = la.norm(x - full) / la.norm(full)
            psnr = 10 * math.log10( n1 * n2 * (255 ** 2) / la.norm(x - full) ** 2  )
            resultlist.append(psnr)
        if data == 'random':
            relative_error = la.norm(x - full) / la.norm(full)
            resultlist.append(relative_error)
        if data == 'movielens':
            relative_error = la.norm(x_omega_ndarray  - M1) / (len(M1.nonzero()[0])* 4)
            resultlist.append(relative_error)
        diff = P_Omega_M - x_omega
        Y += delta * diff


    if data == 'lena' or data =='noisy' or data == 'picture' or data == 'noisy_picture':
        image = MatrixToImage(x)
        image.save(r'C:\Users\pc\Desktop\毕设图片\\' + data + '_SVT_experiment+'+str(experiment)+'.png')

    return timelist,resultlist