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

    evaldir = filepath + '/SampleSet_ds_b30_fov120/'
    evaldir_disp = filepath + '/SampleSet_ds_b30_fov120/disp/'

    subdir = ['/cam_60/', '/cam_60_15bs/']
    imm_l = os.listdir(dir + subdir[0])
    for im in imm_l:
        if is_image_file(dir + subdir[0] + im):
            all_left_img.append(dir + subdir[0] + im)

        all_left_disp.append(dir_disp + im)

        if is_image_file(dir + subdir[1] + im):
            all_right_img.append(dir + subdir[1] + im)

    subdir = ['/left/', '/right/']
    imm_l = os.listdir(evaldir + subdir[0])
    for im in imm_l:
        if is_image_file(evaldir + subdir[0] + im):
            eval_left_img.append(evaldir + subdir[0] + im)

        eval_left_disp.append(evaldir_disp + im)

        if is_image_file(evaldir + subdir[1] + im):
            eval_right_img.append(evaldir + subdir[0] + im)

    return all_left_img, all_right_img, all_left_disp, eval_left_img, eval_right_img, eval_left_disp
