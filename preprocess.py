import numpy as np


def createBoxes(images, coordinates):
    seq_length = images.shape[2]
    for n in range(seq_length):
        pixel_start_x = int(round(x1[n]))
        pixel_start_y = int(round(y1[n]))
        pixel_end_x = int(round(x2[n]))
        pixel_end_y = int(round(y2[n]))
        for ri in range(pixel_end_y-pixel_start_y+1):
            for ci in range(pixel_end_x-pixel_start_x+1):
                images[pixel_start_y+ri, pixel_start_x+ci, n] = 1
    return images

