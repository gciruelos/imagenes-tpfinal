import numpy as np
import scipy.fftpack
from PIL import Image
from sys import argv
import math
import cv2

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

M = 8
Q = 4
U = 256

def block_division(img):
    blocks = []
    k1, k2 = img.shape
    for start1 in range(k1 // M):
        for start2 in range(k2 // M):
            block = np.zeros((M,M), dtype=np.uint8)
            for i in range(M):
                for j in range(M):
                    block[i,j] = img[start1 * M + i, start2 * M + j]
            blocks.append(block)
    return blocks

def apply_dct(block):
    #return scipy.fftpack.dct(block)
    '''
    dct_block = np.zeros((M,M), dtype=np.float)

    c = lambda x: 1./np.sqrt(2) if x == 0 else 1.

    for u in range(M):
        for v in range(M):
            f_uv = c(u) * c(v) * 0.25
            accum = 0
            for x in range(M):
                for y in range(M):
                    f_xy = float(block[x,y]) - U // 2 + 1
                    accum += f_xy * np.cos(np.pi * u * (2 * x + 1) / 16.) * np.cos(np.pi * v * (2 * y + 1) / 16.)
            dct_block[u,v] = f_uv * accum
    return dct_block
    '''
    #return cv2.dct(np.float32(block) - U // 2 + 1)
    return cv2.dct(np.float32(block))




def quantize(dct_block, q):
    block = np.zeros((M,M), dtype=np.int)
    k1, k2 = dct_block.shape
    for i in range(k1):
        for j in range(k2):
            block[i,j] = dct_block[i,j] / q
    return block






image = np.asarray(Image.open(argv[1]).convert('L'))

image_blocks = block_division(image)
dct_blocks = list(map(apply_dct, image_blocks))
print(dct_blocks)
#quantized_blocks = list(map(lambda x: quantize(x, Q), dct_blocks))
#print(quantized_blocks)


