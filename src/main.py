import numpy as np
import scipy.fftpack
from PIL import Image
from sys import argv
import math
import cv2

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

M = 8
Q = 32
U = 256

###############################################################################
###########################  FORWARD  #########################################
###############################################################################

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
    #return cv2.dct(np.float32(block) - U // 2 + 1)
    return cv2.dct(np.float32(block))

def quantize(dct_block, q):
    block = np.zeros((M,M), dtype=np.int)
    k1, k2 = dct_block.shape
    for i in range(k1):
        for j in range(k2):
            block[i,j] = dct_block[i,j] / q
    return block

def compress_dc(uncompressed):
    for i in range(len(uncompressed)-1, 0, -1):
        uncompressed[i][0,0] -= uncompressed[i-1][0,0]

###############################################################################
###########################  INVERSE  #########################################
###############################################################################

def join_blocks(blocks):
    image_size = int(np.sqrt(len(blocks)) * M)
    full_image = np.zeros((image_size,image_size), dtype=np.int)
    for b in range(len(blocks)):
        for i in range(M):
            for j in range(M):
                full_image[(b // (image_size // M)) * M + i, (b % (image_size // M)) * M + j]  = blocks[b][i,j]
    return full_image

def apply_idct(block):
    return cv2.idct(np.float32(block))

def unquantize(block, q):
    dct_block = np.zeros((M,M), dtype=np.float32)
    k1, k2 = dct_block.shape
    for i in range(k1):
        for j in range(k2):
            dct_block[i,j] = block[i,j] * q
    return dct_block

def uncompress_dc(compressed):
    for i in range(1, len(compressed)):
        compressed[i][0,0] += compressed[i-1][0,0]

image = np.asarray(Image.open(argv[1]).convert('L'))

##### FORWARD ###########
image_blocks = block_division(image)
dct_blocks = list(map(apply_dct, image_blocks))
quantized_blocks = list(map(lambda x: quantize(x, Q), dct_blocks))
#print(quantized_blocks)
compress_dc(quantized_blocks)
print(quantized_blocks)
##### INVERSE ###########
uncompress_dc(quantized_blocks)
unquantized_blocks = list(map(lambda x: unquantize(x, Q), quantized_blocks))
original_blocks = list(map(apply_idct, unquantized_blocks))
restored_image = join_blocks(original_blocks)



plt.subplot(1,2,1).imshow(image, cmap='gray') #, vmin=0, vmax=255)
plt.subplot(1,2,2).imshow(restored_image, cmap='gray') #, vmin=0, vmax=255)
plt.show()


