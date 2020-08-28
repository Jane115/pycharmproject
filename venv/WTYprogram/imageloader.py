import os
import numpy as np
import scipy as sp
from PIL import Image
import scipy.sparse as ss
import matplotlib.pyplot as plt


def pictureloader(sp,rank):
    im = Image.open('dataset/picture.png')
    height, width = im.size
    gray = im.convert('L')
    original_matrix_ndarray = np.array(gray)
    low_rank_matrix_ndarray = low_rank_approximation(original_matrix_ndarray, rank)
    mask = np.random.choice([0, 1], (width, height), p=[(1 - sp), sp])
    masked_matrix_ndarray = mask * low_rank_matrix_ndarray
    return masked_matrix_ndarray, mask, low_rank_matrix_ndarray



def MatrixToImage(data):
    #data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im

def low_rank_approximation(X, rank):
    # 奇异值分解
    U, d, Vt = sp.linalg.svd(X)
    d[rank:] = 0
    #Construct the sigma matrix in SVD from singular values and size M, N.
    D = sp.linalg.diagsvd(d, X.shape[0], X.shape[1])
    return np.dot(np.dot(U, D), Vt)


def imageloader(sp,rank):
    # 读取图片
    # the type of loaded image is PIL.JpegImagePlugin.JpegImageFile
    im = Image.open('dataset/lena.png')
    height, width  = im.size
    data = im.getdata()
    original_matrix_ndarray = np.reshape(data, (width, height))
    low_rank_matrix_ndarray = low_rank_approximation(original_matrix_ndarray,rank)

    mask = np.random.choice([0, 1], (width, height), p=[(1-sp), sp])
    masked_matrix_ndarray = mask * low_rank_matrix_ndarray
    return masked_matrix_ndarray, mask, low_rank_matrix_ndarray

def imageloader_noisy(sp,rank):
    # the type of loaded image is PIL.JpegImagePlugin.JpegImageFile
    im = Image.open('dataset/lena.png')
    height, width  = im.size
    data = im.getdata()
    original_matrix_ndarray = np.reshape(data, (width, height))

    low_rank_matrix_ndarray = low_rank_approximation(original_matrix_ndarray,rank)
    noise = np.random.randn(width, height)
    noisy_matrix = low_rank_matrix_ndarray + noise
    mask = np.random.choice([0, 1], (width, height), p=[(1-sp), sp])
    masked_matrix_ndarray = mask * noisy_matrix
    return masked_matrix_ndarray, mask, low_rank_matrix_ndarray

def picutreloader_noisy(sp,rank):
    # the type of loaded image is PIL.JpegImagePlugin.JpegImageFile
    im = Image.open('dataset/picture.png')
    height, width  = im.size
    gray = im.convert('L')
    original_matrix_ndarray = np.array(gray)

    low_rank_matrix_ndarray = low_rank_approximation(original_matrix_ndarray,rank)
    noise = np.random.randn(width, height)
    noisy_matrix = low_rank_matrix_ndarray + noise
    mask = np.random.choice([0, 1], (width, height), p=[(1-sp), sp])
    masked_matrix_ndarray = mask * noisy_matrix
    return masked_matrix_ndarray, mask, low_rank_matrix_ndarray



