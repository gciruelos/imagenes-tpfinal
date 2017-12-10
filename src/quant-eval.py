import main
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
import math

def psnr(img, compressed_img, is_rgb):
    mse = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not is_rgb:
                mse += math.pow(float(float(img[i][j]) - float(compressed_img[i][j])), 2)
            else:
                mse += math.pow(float(float(img[i][j][0]) - float(compressed_img[i][j][0])), 2)
                mse += math.pow(float(float(img[i][j][1]) - float(compressed_img[i][j][1])), 2)
                mse += math.pow(float(float(img[i][j][2]) - float(compressed_img[i][j][2])), 2)
    mse /= (img.shape[0] * img.shape[1] * (3 if is_rgb else 1))
    max_i = 255.0 * 255.0 * (3 if is_rgb else 1)
    return 10 * math.log10(max_i / mse)

def plot_comparison(ticks, psnr, compression, xlabel, ftitle):
    fig, ax1 = plt.subplots()
    fig.set_size_inches(8, 6)
    ax1.plot(ticks, psnr, 'b-')
    ax1.set_xlabel(xlabel)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('PSNR', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(ticks, compression, 'r-')
    ax2.set_ylabel('Compresion', color='r')
    ax2.tick_params('y', colors='r')

    # fig.tight_layout()
    plt.title(ftitle)
    if len(argv) > 3:
        plt.savefig(argv[3])
    else:
        plt.show()


def test_config(image, config, config_desc, ticks, is_rgb):
    psnrl = []
    compressionl = []
    image_size = (image.shape[0] * image.shape[1] * (3 if is_rgb else 1))
    title = argv[1].split('/')[-1]+' '
    m_title = 'Bloque: {0}, '.format(main.M)
    mcolor_title = 'Bloque color: {0}, '.format(main.M_COLOR)
    qac_title = 'Cuant. AC: {0}, '.format(main.Q_AC)
    qdc_title = 'Cuant. DC: {0}, '.format(main.Q_DC)
    u_title = 'Threshold cuant.: {0}, '.format(main.U)
    next_title = ''
    if config == 'M':
        next_title = (mcolor_title if is_rgb else '')+qac_title+qdc_title+u_title
    if config == 'Q_AC':
        next_title = m_title+(mcolor_title if is_rgb else '')+qdc_title+u_title
    if config == 'Q_DC':
        next_title = m_title+(mcolor_title if is_rgb else '')+qdc_title+u_title
    if config == 'M_COLOR':
        next_title = m_title+qac_title+qdc_title+u_title
    if config == 'U':
        next_title = m_title+(mcolor_title if is_rgb else '')+qdc_title+qac_title
    title += ('(Tamanio: {0} KB) {1}'.format(int(image_size / 1024), next_title[:-2]))
    for t in ticks:
        if config == 'M':
            main.M = t
        elif config == 'M_COLOR':
            main.M_COLOR = t
        elif config == 'Q_AC':
            main.Q_AC = t
        elif config == 'Q_DC':
            main.Q_DC = t
        elif config == 'U':
            main.U = t
        huffman_coding, restored_image = main.greyscale(image) if not is_rgb else main.rgb(image)
        psnrl.append(psnr(image, restored_image, is_rgb))
        compressionl.append(huffman_coding / image_size)
    plot_comparison(ticks, psnrl, compressionl, config_desc, title)


configs = {
    'M' : ('Tamano del bloque', [2,4,8,16,32,64,128]),
    'M_COLOR' : ('Tamano del bloque de color', [1,2,4,8,16,32]),
    'Q_DC' : ('Cuantizacion de DC', [0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5]),
    'Q_AC' : ('Cuantizacion de AC', [0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.5]),
    'U' : ('Threshold de cuantizacion', [1,2, 2.5, 2.8, 3, 3.5, 4, 5 ,6])
    }

test__ = argv[2]
desc__ = configs[test__][0]
ticks__ = configs[test__][1]


if __name__ == '__main__':
    pil_image = Image.open(argv[1])
    if main.is_greyscale(pil_image):
        image = np.asarray(pil_image.convert('L'))
        test_config(image, test__, desc__, ticks__, False)
    else:
        image = np.asarray(pil_image.convert('RGB'))
        test_config(image, test__, desc__, ticks__, True)
