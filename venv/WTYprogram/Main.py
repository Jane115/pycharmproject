# import packages
import sys
import cv2
import math
from AM import *
from SVT import *
from ASD import *
from SVP import *
from Schatten import *
import numpy as np
from scipy import misc
from dataloader import *
from imageloader import *
from time import perf_counter
from numpy import linalg as la
import matplotlib.pyplot as plt
from timeit import default_timer as timer

Experiment = 9
plot = True

if Experiment == 1:
    data = 'random'
    matrixsize = 100
    rank = 5
    sp = 0.5
    iter_SVT = 2000
    iter_SVP = 50
    iter_ASD = 50
    iter_Schatten = 5

elif Experiment == 2:
    data = 'random'
    matrixsize = 1000
    rank = 5
    sp = 0.5
    iter_SVT = 80
    iter_SVP = 20
    iter_ASD = 10
    iter_Schatten = 3

elif Experiment == 3:
    data = 'random'
    matrixsize = 100
    rank = 100
    sp = 0.5
    iter_SVT = 200
    iter_SVP = 50
    iter_ASD = 50
    iter_Schatten = 5

elif Experiment == 4:
    data = 'random'
    matrixsize = 100
    rank = 5
    sp = 0.8
    iter_SVT = 1000
    iter_SVP = 20
    iter_ASD = 50
    iter_Schatten = 5

elif Experiment == 5:
    data = 'lena'
    rank = 50
    sp = 0.5
    iter_SVT = 400
    iter_SVP = 1300
    iter_ASD = 200
    iter_Schatten = 3

elif Experiment == 6:
    data = 'noisy'
    rank = 50
    sp = 0.5
    iter_SVT = 110
    iter_SVP = 600
    iter_ASD = 100
    iter_Schatten = 3

elif Experiment == 7:
    data = 'movielens'
    matrixsize = 500
    rank = 20
    iter_SVT = 1000
    iter_SVP = 3
    iter_ASD = 200
    iter_Schatten = 2

elif Experiment == 8:
    data = 'picture'
    rank = 50
    sp = 0.5
    iter_SVT = 400
    iter_SVP = 1300
    iter_ASD = 200
    iter_Schatten = 3

elif Experiment == 9:
    data = 'noisy_picture'
    rank = 50
    sp = 0.5
    iter_SVT = 110
    iter_SVP = 600
    iter_ASD = 100
    iter_Schatten = 3



# main method
if __name__ == '__main__':

    # data loading
    if data  == 'movielens':
        masked_matrix_ndarray, mask, original_matrix_ndarray = dataloader_movielens(matrixsize)
        masked_matrix_sparse = ss.csr_matrix(masked_matrix_ndarray)
        sp = len(masked_matrix_ndarray.nonzero()[0])/(matrixsize * matrixsize)


    elif data == 'random':
        masked_matrix_ndarray, mask, original_matrix_ndarray = dataloader_random(matrixsize, sp, rank)
        masked_matrix_sparse = ss.csr_matrix(masked_matrix_ndarray)


    elif data == 'lena':
        masked_matrix_ndarray, mask, original_matrix_ndarray = imageloader(sp,rank)
        masked_matrix_sparse = ss.csr_matrix(masked_matrix_ndarray)

    elif data == 'noisy':
        masked_matrix_ndarray, mask, original_matrix_ndarray = imageloader_noisy (sp, rank)
        masked_matrix_sparse = ss.csr_matrix(masked_matrix_ndarray)

    elif data == 'picture':
        masked_matrix_ndarray, mask, original_matrix_ndarray = pictureloader (sp, rank)
        masked_matrix_sparse = ss.csr_matrix(masked_matrix_ndarray)

    elif data == 'noisy_picture':
        masked_matrix_ndarray, mask, original_matrix_ndarray = picutreloader_noisy (sp, rank)
        masked_matrix_sparse = ss.csr_matrix(masked_matrix_ndarray)



    timelist_SVT = []
    resultlist_SVT = []
    timelist_SVP = []
    resultlist_SVP = []
    timelist_ASD = []
    resultlist_ASD = []
    timelist_Schatten = []
    resultlist_Schatten = []

    # timelist_SVT, resultlist_SVT = SVT(masked_matrix_sparse, original_matrix_ndarray,iter_SVT,data,Experiment)
    # timelist_SVP, resultlist_SVP = SVP(masked_matrix_ndarray, original_matrix_ndarray, mask, rank, sp, iter_SVP, data,Experiment)
    timelist_ASD, resultlist_ASD = ASD(masked_matrix_ndarray, original_matrix_ndarray, mask, rank, iter_ASD, data,Experiment)
    # timelist_Schatten, resultlist_Schatten = OtraceEEC(masked_matrix_ndarray, mask, original_matrix_ndarray,iter_Schatten,data,Experiment)
    # print(resultlist_SVT[-1],resultlist_SVP[-1], resultlist_ASD[-1],resultlist_Schatten[-1])
    print(resultlist_ASD[-1])



    if plot:
        # #rmse画图
        # plt.figure(1)
        # plt.plot(timelist_SVT, resultlist_SVT, '-', label='SVT')
        # plt.title('Performance of Singular Value Thresholding')
        # plt.legend()
        # plt.grid()
        # plt.xlabel('Time Elapse (s)')
        # if data == 'random':
        #     plt.ylabel('Relative Error')
        # elif data in ['lena', 'picture', 'noisy', 'noisy_picture']:
        #     plt.ylabel('Peak Signal to Noise Ratio (PSNR)')
        # elif data == 'movielens':
        #     plt.ylabel('Normalized Mean Absolute Error(NMAE)')
        # #plt.show()
        # plt.savefig(r'C:\Users\pc\Desktop\毕设图片\experiment'+str(Experiment)+'_SVT.png', dpi=300)
        #
        # plt.figure(2)
        # plt.plot(timelist_SVP, resultlist_SVP, '-', label='SVP')
        # plt.title('Performance of Singular Value Projection')
        # plt.legend()
        # plt.grid()
        # plt.xlabel('Time Elapse (s)')
        # if data == 'random':
        #     plt.ylabel('Relative Error')
        # elif data in ['lena', 'picture', 'noisy', 'noisy_picture']:
        #     plt.ylabel('Peak Signal to Noise Ratio (PSNR)')
        # elif data == 'movielens':
        #     plt.ylabel('Normalized Mean Absolute Error(NMAE)')
        # #plt.show()
        # plt.savefig(r'C:\Users\pc\Desktop\毕设图片\experiment'+str(Experiment)+'_SVP.png', dpi=300)

        plt.figure(3)
        plt.plot(timelist_ASD, resultlist_ASD, '-', label='ASD')
        plt.title('Performance of Alternating Steepest Descent')
        plt.legend()
        plt.grid()
        plt.xlabel('Time Elapse (s)')
        if data == 'random':
            plt.ylabel('Relative Error')
        elif data in ['lena','picture','noisy','noisy_picture']:
            plt.ylabel('Peak Signal to Noise Ratio (PSNR)')
        elif data == 'movielens':
            plt.ylabel('Normalized Mean Absolute Error(NMAE)')
        #plt.show()
        plt.savefig(r'C:\Users\pc\Desktop\毕设图片\experiment'+str(Experiment)+'_ASD.png', dpi=300)


        # plt.figure(4)
        # plt.plot(timelist_Schatten, resultlist_Schatten, '-', label='Schatten-p')
        # plt.title('Performance of Schatten-p Minimization')
        # #plt.xlim(right = 4)
        # plt.legend()
        # plt.grid()
        # plt.xlabel('Time Elapse (s)')
        # if data == 'random':
        #     plt.ylabel('Relative Error')
        # elif data in ['lena', 'picture', 'noisy', 'noisy_picture']:
        #     plt.ylabel('Peak Signal to Noise Ratio (PSNR)')
        # elif data == 'movielens':
        #     plt.ylabel('Normalized Mean Absolute Error(NMAE)')
        # #plt.show()
        # plt.savefig(r'C:\Users\pc\Desktop\毕设图片\experiment'+str(Experiment)+'_Schatten.png', dpi=300)



