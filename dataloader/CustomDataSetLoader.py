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

    #data_dirs = ['/20210901-1211_x/', '/20210901-1216_y/', '/20210901-1221_z/', '/20210901-1232_xyz/']

    dir = filepath
    dir_disp = filepath + '/cam_dep_60_Bl0/'

    subdir = ['/cam_60_BL0/', '/cam_60_BL30/']#, '/cam_60_BL30_ch_x/', '/cam_60_BL30_ch_y/', '/cam_60_BL30_ch_z/', '/cam_60_BL30_ch_yaw/']

    #train_file = open(os.path.join(dir, 'train.txt'), 'r')
    #valid_file = open(os.path.join(dir, 'valid.txt'), 'r')

    images = [img for img in os.listdir(filepath + subdir[0])]

    train_lines = images[:200]
    valid_lines = images[200:]

    #train_lines = train_file.readlines()
    #valid_lines = valid_file.readlines()

    for line in train_lines:
        #line = line.strip()

        all_left_img.append(dir + subdir[0] + line)
        all_left_disp.append(dir_disp + line)
        all_right_img.append(dir + subdir[1] + line)

        #all_left_img.append(dir + subdir[0] + line)
        #all_left_disp.append(dir_disp + line)
        #all_right_img.append(dir + subdir[2] + line)

        #all_left_img.append(dir + subdir[0] + line)
        #all_left_disp.append(dir_disp + line)
        #all_right_img.append(dir + subdir[3] + line)

        #all_left_img.append(dir + subdir[0] + line)
        #all_left_disp.append(dir_disp + line)
        #all_right_img.append(dir + subdir[4] + line)

        #all_left_img.append(dir + subdir[0] + line)
        #all_left_disp.append(dir_disp + line)
        #all_right_img.append(dir + subdir[5] + line)

    for line in valid_lines:
        #line = line.strip()

        eval_left_img.append(dir + subdir[0] + line)
        eval_left_disp.append(dir_disp + line)
        eval_right_img.append(dir + subdir[1] + line)

        #eval_left_img.append(dir + subdir[0] + line)
        #eval_left_disp.append(dir_disp + line)
        #eval_right_img.append(dir + subdir[2] + line)

        #eval_left_img.append(dir + subdir[0] + line)
        #eval_left_disp.append(dir_disp + line)
        #eval_right_img.append(dir + subdir[3] + line)

        #eval_left_img.append(dir + subdir[0] + line)
        #eval_left_disp.append(dir_disp + line)
        #eval_right_img.append(dir + subdir[4] + line)

        #eval_left_img.append(dir + subdir[0] + line)
        #eval_left_disp.append(dir_disp + line)
        #eval_right_img.append(dir + subdir[5] + line)

    

    return all_left_img, all_right_img, all_left_disp, eval_left_img, eval_right_img, eval_left_disp
