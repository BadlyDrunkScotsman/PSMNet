import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    dir_disp = filepath + '/cam_dep_60_left/'

    subdir = ['/cam_60_left/', '/cam_60_right/']

    
    image = [img for img in os.listdir(filepath+subdir[0])]

    left_train  = [filepath+subdir[0]+img for img in image]
    right_train = [filepath+subdir[1]+img for img in image]
    disp_train_L = [dir_disp+img for img in image]

    return left_train, right_train, disp_train_L, left_train, right_train, disp_train_L
