import numpy as np
import scipy.fftpack
from PIL import Image
from sys import argv
import math
import cv2
import huffman
import collections

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

M = 8
Q = 16
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

def entropy_coding(block):
    zigzag = []
    x = 0
    y = 0
    for s in range(2 *M-1):
        for t in range(0, M):
            if 0 <= s - t < M:
                if s % 2 == 1: zigzag.append(block[t,s-t])
                else: zigzag.append(block[s-t, t])
    compressed_zigzag = []
    i = 0
    while i < len(zigzag):
        zero_preceding = 0
        while i < len(zigzag) and zigzag[i] == 0:
            zero_preceding += 1
            i += 1
        else:
            if i >= len(zigzag):
                break
        compressed_zigzag.append((zigzag[i], zero_preceding))
        i += 1
    return compressed_zigzag

def huffman_code(zigzag_blocks):
    total_numbers = []
    for zz in zigzag_blocks:
        total_numbers += zz
    h = huffman.codebook(collections.Counter(total_numbers).items())
    string_result = ''
    for n in total_numbers:
        string_result += h[n]
    return string_result, h


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

###############################################################################
#################################  RGB  #######################################
###############################################################################

def block_division_rgb(img):
    blocks = []
    crcb_blocks = []
    k1, k2, k3 = img.shape
    for start1 in range(k1 // M):
        for start2 in range(k2 // M):
            block = np.zeros((M,M), dtype=np.uint8)
            cr = 0.0
            cb = 0.0
            for i in range(M):
                for j in range(M):
                    block[i,j] = img[start1 * M + i, start2 * M + j, 0]
                    cr += img[start1 * M + i, start2 * M + j, 1]
                    cb += img[start1 * M + i, start2 * M + j, 2]
            blocks.append(block)
            crcb_blocks.append((int(cr / (M*M)), int(cb / (M*M))))
    return blocks, crcb_blocks

def join_blocks_rgb(blocks, crcb_blocks):
    image_size = int(np.sqrt(len(blocks)) * M)
    full_image = np.zeros((image_size,image_size,3), dtype=np.int)
    for b in range(len(blocks)):
        for i in range(M):
            for j in range(M):
                full_image[(b // (image_size // M)) * M + i, (b % (image_size // M)) * M + j,:] = [blocks[b][i,j],crcb_blocks[b][0], crcb_blocks[b][1]]
    return full_image

###############################################################################
#########################################  MAIN ###############################
###############################################################################

def is_greyscale(image):
    rgb_image = image.convert('RGB')
    w,h = rgb_image.size
    for i in range(w):
        for j in range(h):
            r,g,b = rgb_image.getpixel((i,j))
            if r != g != b: return False
    return True

def greyscale(image):
    ##### FORWARD ###########
    image_blocks = block_division(image)
    dct_blocks = list(map(apply_dct, image_blocks))
    quantized_blocks = list(map(lambda x: quantize(x, Q), dct_blocks))
    compress_dc(quantized_blocks)
    zigzag_blocks = list(map(entropy_coding, quantized_blocks))
    huffman_coding, huffman_map = huffman_code(zigzag_blocks)

    ##### INVERSE ###########
    uncompress_dc(quantized_blocks)
    unquantized_blocks = list(map(lambda x: unquantize(x, Q), quantized_blocks))
    original_blocks = list(map(apply_idct, unquantized_blocks))
    restored_image = join_blocks(original_blocks)
    return huffman_coding, restored_image

def rgb(image):
    ##### FORWARD ###########
    print(image)
    image_ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    image_blocks, cbcr_blocks = block_division_rgb(image)
    dct_blocks = list(map(apply_dct, image_blocks))
    quantized_blocks = list(map(lambda x: quantize(x, Q), dct_blocks))
    compress_dc(quantized_blocks)
    zigzag_blocks = list(map(entropy_coding, quantized_blocks))
    huffman_coding, huffman_map = huffman_code(zigzag_blocks)

    ##### INVERSE ###########
    uncompress_dc(quantized_blocks)
    unquantized_blocks = list(map(lambda x: unquantize(x, Q), quantized_blocks))
    original_blocks = list(map(apply_idct, unquantized_blocks))
    restored_image = join_blocks_rgb(original_blocks, cbcr_blocks)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_YCR_CB2RGB)
    print(image_rgb)
    return huffman_coding, image_rgb



pil_image = Image.open(argv[1])
if is_greyscale(pil_image):
    image = np.asarray(pil_image.convert('L'))
    huffman_coding, restored_image = greyscale(image)
    print(huffman_coding)
    print('Compressed size:', len(huffman_coding) / 1024, 'KB')
    plt.subplot(1,2,1).imshow(image, cmap='gray')
    plt.subplot(1,2,2).imshow(restored_image, cmap='gray')
    plt.show()
else:
    image = np.asarray(pil_image.convert('RGB'))
    huffman_coding, restored_image = rgb(image)
    print(huffman_coding)
    print('Compressed size:', len(huffman_coding) / 1024, 'KB')
    plt.subplot(1,2,1).imshow(image)
    plt.subplot(1,2,2).imshow(restored_image)
    plt.show()


