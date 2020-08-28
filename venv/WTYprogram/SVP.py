import time
from ASD import *
import numpy as np
import scipy as sp
from scipy import linalg
from imageloader import *
from numpy.linalg import norm
from numpy import linalg as la



def SVP(z0, full,  mask, rank, p, max_iter,data,experiment):
    print('SVP begins')
    delta  = 0.2
    m, n = z0.shape
    X = np.zeros((m,n))
    resultlist = []
    timelist = []
    # ranks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # rank_result = []
    # for rank in ranks:
    start_time = time.perf_counter()
    for i in range(max_iter):
        print('SVP iteration:', i)
        # 奇异值分解
        U, d, Vt = sp.linalg.svd(X - (1/((1+delta)*p)*(X * mask - z0)))


        d[rank:] = 0
        # Construct the sigma matrix in SVD from singular values and size M, N.
        D = sp.linalg.diagsvd(d, X.shape[0], X.shape[1])
        X =  np.dot(np.dot(U, D), Vt)
        mask_X = X * mask
        end_time = time.perf_counter()
        timelist.append(end_time - start_time)

        if data == 'lena' or data == 'noisy' or data == 'picture' or data == 'noisy_picture':
            relative_error = la.norm(X - full) / la.norm(full)

            psnr = 10 * math.log10( m * n * (255 ** 2) / la.norm(X - full) ** 2  )
            resultlist.append(psnr)
        if data == 'random':
            relative_error = la.norm(X - full) / la.norm(full)
            resultlist.append(relative_error)

        if data == 'movielens':
            relative_error = la.norm(mask_X  - z0) / (len(z0.nonzero()[0])* 4)
            resultlist.append(relative_error)


    if data == 'lena' or data =='noisy' or data == 'picture' or data == 'noisy_picture':
        image = MatrixToImage(X)
        image.save(r'C:\Users\pc\Desktop\毕设图片\\' + data + '_SVP_experiment+'+str(experiment)+'.png')

    return timelist, resultlist