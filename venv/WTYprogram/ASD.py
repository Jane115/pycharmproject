#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import time
import math
import numpy as np
import scipy as sp
from scipy import misc
from scipy import linalg
from imageloader import *
from numpy.linalg import norm
from numpy import linalg as la
import matplotlib.pyplot as plt


def ASD(z0, full,  mask, rank, max_iter,data,experiment):
    print('ASD begins')
    #两个初始化矩阵
    m,n = z0.shape

    timelist = []
    resultlist = []
    # rank_result = []
    # ranks = [10,20,30,40,50,60,70,80,90,100]

    x = np.random.randint(0, 255, (m, rank))

    # x0 值为 0-255 dimension r * n
    y = np.random.randint(0, 255, (rank, n))

    # 矩阵向量乘法
    xy = np.dot(x,y)
    diff = mask * (z0 - xy)

    start_time = time.perf_counter()
    for num_iter in range(max_iter):
        print('ASD iteration:', num_iter)
        grad_x = np.dot(-diff , y.T)

        delta_xy = mask * np.dot(grad_x , y)

        tx = norm(grad_x) ** 2 / norm(delta_xy) ** 2
        x = x - tx * grad_x

        diff = diff + tx * delta_xy

        grad_y = np.dot(-x.T , diff)


        delta_xy = mask * np.dot(x , grad_y)
        ty = norm(grad_y) ** 2 / norm(delta_xy) ** 2
        y = y - ty * grad_y

        diff = diff + ty * delta_xy
        #diff = mask * (z0 - xy)
        xy = np.dot(x,y)
        mask_xy = xy * mask

        if data == 'lena' or data == 'noisy' or data == 'picture' or data == 'noisy_picture':
            end_time = time.perf_counter()
            timelist.append(end_time - start_time)
            relative_error = la.norm(xy - full) / la.norm(full)
            psnr = 10 * math.log10( m * n * (255 ** 2) / la.norm(xy - full) ** 2  )
            resultlist.append(psnr)
        if data == 'random':
            relative_error = la.norm(xy - full) / la.norm(full)
            if experiment in [1,2]:
                if relative_error < 2:
                    end_time = time.perf_counter()
                    timelist.append(end_time - start_time)
                    resultlist.append(relative_error)
            else:
                end_time = time.perf_counter()
                timelist.append(end_time - start_time)
                resultlist.append(relative_error)

        if data == 'movielens':
            end_time = time.perf_counter()
            timelist.append(end_time - start_time)
            relative_error = la.norm(mask_xy  - z0) / (len(z0.nonzero()[0])* 4)
            resultlist.append(relative_error)

    if data == 'lena' or data =='noisy' or data == 'picture' or data == 'noisy_picture':
        image = MatrixToImage(xy)
        image.save(r'C:\Users\pc\Desktop\毕设图片\\' + data + '_ASD_experiment+'+str(experiment)+'.png')

    return timelist,resultlist


