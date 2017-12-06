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

M = 32
M_COLOR = 4
Q_AC = 3
Q_DC = 3
U = 1

###############################################################################
###########################  FORWARD  #########################################
###############################################################################

jpeg_table = [
[16, 11 ,10 ,16 ,24 ,40  , 51 , 61     ],
[12, 12 ,14 ,19 ,26 ,58  , 60 , 55     ],
[14, 13 ,16 ,24 ,40 ,57  , 69 , 56     ],
[14, 17 ,22 ,29 ,51 ,87  , 80 , 62     ],
[18, 22 ,37 ,56 ,68 ,109 , 103, 77   ],
[24, 35 ,55 ,64 ,81 ,104 , 113, 92   ],
[49, 64 ,78 ,87 ,103, 121, 120, 101 ],
[72, 92 ,95 ,98 ,112, 100, 103, 99  ]
]

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

def quantize(dct_block):
    block = np.zeros((M,M), dtype=np.int)
    k1, k2 = dct_block.shape
    for i in range(k1):
        for j in range(k2):
            q = Q_DC if i == 0 and j == 0 else Q_AC
            block[i,j] = dct_block[i,j] / (jpeg_table[i // (8 * M)][j // (8 * M)] * q)
            if abs(block[i,j]) < U:
                block[i,j] = 0
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

def join_blocks(blocks, image):
    imsize = np.shape(image)
    full_image = np.zeros(imsize, dtype=np.uint8)
    k1, k2 = (k // M for k in imsize[:2])
    for x in range(k1):
        for y in range(k2):
            b = x * k2 + y
            full_image[x * M : (x + 1) * M, y * M : (y + 1) * M] = blocks[b]
    full_image[:, k2 * M:] = image[:, k2 * M:]
    full_image[k1 * M:, :k2 * M] = image[k1 * M:, :k2 * M]
    return full_image

def apply_idct(block):
    return cv2.idct(np.float32(block))

def unquantize(block):
    dct_block = np.zeros((M,M), dtype=np.float32)
    k1, k2 = dct_block.shape
    for i in range(k1):
        for j in range(k2):
            q = Q_DC if i == 0 and j == 0 else Q_AC
            dct_block[i,j] = block[i,j] * q * jpeg_table[i // (8 * M)][j // (8 * M)]
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
    k1, k2 = (img.shape[0] // M, img.shape[1] // M)
    for start1 in range(k1):
        for start2 in range(k2):
            block = np.zeros((M,M), dtype=np.uint8)
            cr = 0.0
            cb = 0.0
            for i in range(M):
                for j in range(M):
                    block[i,j] = img[start1 * M + i, start2 * M + j, 0]
            blocks.append(block)
    k1, k2 = (img.shape[0] // M_COLOR, img.shape[1] // M_COLOR)
    for start1 in range(k1):
        for start2 in range(k2):
            cr = 0.0
            cb = 0.0
            for i in range(M_COLOR):
                for j in range(M_COLOR):
                    cr += img[start1 * M_COLOR + i, start2 * M_COLOR + j, 1]
                    cb += img[start1 * M_COLOR + i, start2 * M_COLOR + j, 2]
            crcb_blocks.append((int(cr / (M_COLOR*M_COLOR)), int(cb / (M_COLOR*M_COLOR))))
    return blocks, crcb_blocks

def join_blocks_rgb(blocks, crcb_blocks, img):
    imsize = np.shape(img)
    full_image = np.zeros(imsize, dtype=np.uint8)
    k1, k2 = (k // M for k in imsize[:2])
    for x in range(k1):
        for y in range(k2):
            for i in range(M):
                for j in range(M):
                    b = x * k2 + y
                    full_image[x * M + i, y * M + j,0] = blocks[b][i,j]
    kC1, kC2 = (k // M_COLOR for k in imsize[:2])
    for x in range(kC1):
        for y in range(kC2):
            for i in range(M_COLOR):
                for j in range(M_COLOR):
                    b = x * kC2 + y
                    full_image[x * M_COLOR + i, y * M_COLOR + j,1:] = [crcb_blocks[b][0], crcb_blocks[b][1]]
    #for x in range(imsize[0]):
    #    for y in range(k2 * M, imsize[1]):
    #        full_image[x, y] = img[x, y]
    #for x in range(k1 * M, imsize[0]):
    #    for y in range(k2 * M):
    #        full_image[x, y] = img[x, y]
    for x in range(imsize[0]):
        for y in range(min(k2 * M, kC2 * M_COLOR), imsize[1]):
            full_image[x, y] = img[x, y]
    for x in range(min(k1 * M, kC1 * M_COLOR), imsize[0]):
        for y in range(min(k2 * M, kC2 * M_COLOR)):
            full_image[x, y] = img[x, y]
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
    quantized_blocks = list(map(quantize, dct_blocks))
    compress_dc(quantized_blocks)
    zigzag_blocks = list(map(entropy_coding, quantized_blocks))
    huffman_coding, huffman_map = huffman_code(zigzag_blocks)

    ##### INVERSE ###########
    uncompress_dc(quantized_blocks)
    unquantized_blocks = list(map(unquantize, quantized_blocks))
    original_blocks = list(map(apply_idct, unquantized_blocks))
    restored_image = join_blocks(original_blocks, image)
    return huffman_coding, restored_image

def rgb(image):
    ###### FORWARD ###########
    imsize = image.shape
    image_ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    image_blocks, cbcr_blocks = block_division_rgb(image_ycc)
    dct_blocks = list(map(apply_dct, image_blocks))
    quantized_blocks = list(map(quantize, dct_blocks))
    compress_dc(quantized_blocks)
    zigzag_blocks = list(map(entropy_coding, quantized_blocks))
    huffman_coding, huffman_map = huffman_code(zigzag_blocks)

    ##### INVERSE ###########
    uncompress_dc(quantized_blocks)
    unquantized_blocks = list(map(unquantize, quantized_blocks))
    original_blocks = list(map(apply_idct, unquantized_blocks))
    restored_image = join_blocks_rgb(original_blocks, cbcr_blocks, image_ycc)
    image_rgb = cv2.cvtColor(restored_image, cv2.COLOR_YCR_CB2RGB)
    return len(huffman_coding) + (image_ycc.size * 2 / 3) / (M_COLOR * M_COLOR), image_rgb



pil_image = Image.open(argv[1])
if is_greyscale(pil_image):
    image = np.asarray(pil_image.convert('L'))
    huffman_coding, restored_image = greyscale(image)
    #print(huffman_coding)
    print('Compressed size:', len(huffman_coding) / 1024, 'KB')
    plt.subplot(1,2,1).imshow(image, cmap='gray')
    plt.subplot(1,2,2).imshow(restored_image, cmap='gray')
    plt.show()
else:
    image = np.asarray(pil_image.convert('RGB'))
    compressed_size, restored_image = rgb(image)
    #print(huffman_coding)
    print('Compressed size:', compressed_size / 1024, 'KB')
    plt.subplot(1,2,1).imshow(image)
    plt.subplot(1,2,2).imshow(restored_image)
    plt.show()
