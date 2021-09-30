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
    all_left_img = []
    all_right_img = []
    all_left_disp = []

    eval_left_img = []
    eval_right_img = []
    eval_left_disp = []

    dir = filepath
    dir_disp = filepath + '/cam_dep_60/'

    subdir = ['/cam_60/', '/cam_60_15bs/']

    train_file = open(os.path.join(dir, 'train.txt'), 'r')
    valid_file = open(os.path.join(dir, 'valid.txt'), 'r')

    train_lines = train_file.readlines()
    valid_lines = valid_file.readlines()

    for line in train_lines:
        line = line.strip()

        all_left_img.append(dir + subdir[0] + line)
        all_left_disp.append(dir_disp + line)
        all_right_img.append(dir + subdir[1] + line)

    for line in valid_lines:
        line = line.strip()

        eval_left_img.append(dir + subdir[0] + line)
        eval_left_disp.append(dir_disp + line)
        eval_right_img.append(dir + subdir[1] + line)

    return all_left_img, all_right_img, all_left_disp, eval_left_img, eval_right_img, eval_left_disp
