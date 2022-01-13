import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class TranslactionEntry():
    def __init__(self, frame_num, x, y, z, pitch, yaw, roll):
        self.frame_num = frame_num
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.roll = roll
        self.pitch = pitch


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):

    dir = filepath
    dir_disp = filepath + 'cam_dep_60_Bl0/'

    subdir = ['cam_60_BL0/', 'cam_60_BL30_ch_x/', 'cam_60_BL30_ch_y/', 'cam_60_BL30_ch_z/', 'cam_60_BL30_ch_roll/']
    
    right_transforms = []

    rotation_file = open(os.path.join(dir, 'rotation.txt'), 'r')
    rotation_lines = rotation_file.readlines()

    x = 0
    y = 0
    z = 0
    pitch = 0
    yaw = 0
    roll = 0

    for line in rotation_lines:
        line = line.strip()

        if "X:" in line:
            x = float(line.split("X: ")[1])
        if "Y:" in line:
            y = float(line.split("Y: ")[1])
        if "Z:" in line:
            z = float(line.split("Z: ")[1])
        if "Pitch:" in line:
            pitch = float(line.split("Pitch: ")[1])
        if "Yaw:" in line:
            yaw = float(line.split("Yaw: ")[1])
        if "Roll:" in line:
            roll = float(line.split("Roll: ")[1])
            frame_num = int(line.split(" ")[1]) + 21
            right_transforms.append(TranslactionEntry(frame_num, x, y, z, pitch, yaw, roll))
        
    all_left_img = []
    all_right_img = []
    all_left_disp = []

    eval_left_img = []
    eval_right_img = []
    eval_left_disp = []


    train_file = open(os.path.join(dir, 'train.txt'), 'r')
    valid_file = open(os.path.join(dir, 'valid.txt'), 'r')

    train_lines = train_file.readlines()
    valid_lines = valid_file.readlines()

    for line in train_lines:
        line = line.strip()

        all_left_img.append(dir + subdir[0] + line)
        all_left_disp.append(dir_disp + line)
        all_right_img.append(dir + subdir[1] + line)

        all_left_img.append(dir + subdir[0] + line)
        all_left_disp.append(dir_disp + line)
        all_right_img.append(dir + subdir[2] + line)

        all_left_img.append(dir + subdir[0] + line)
        all_left_disp.append(dir_disp + line)
        all_right_img.append(dir + subdir[3] + line)

        all_left_img.append(dir + subdir[0] + line)
        all_left_disp.append(dir_disp + line)
        all_right_img.append(dir + subdir[4] + line)


    for line in valid_lines:
        line = line.strip()

        eval_left_img.append(dir + subdir[0] + line)
        eval_left_disp.append(dir_disp + line)
        eval_right_img.append(dir + subdir[1] + line)

        eval_left_img.append(dir + subdir[0] + line)
        eval_left_disp.append(dir_disp + line)
        eval_right_img.append(dir + subdir[2] + line)

        eval_left_img.append(dir + subdir[0] + line)
        eval_left_disp.append(dir_disp + line)
        eval_right_img.append(dir + subdir[3] + line)

        eval_left_img.append(dir + subdir[0] + line)
        eval_left_disp.append(dir_disp + line)
        eval_right_img.append(dir + subdir[4] + line)

    return all_left_img, all_right_img, all_left_disp, eval_left_img, eval_right_img, eval_left_disp, right_transforms
