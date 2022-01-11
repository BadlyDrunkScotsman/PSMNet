import random

import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
import os

from utils import preprocess
import numpy as np
from math import cos, cosh, sin
import math
# read the image

def identity_matrix():
    return np.eye(3)

def scaling_matrix(cx, cy):
    return np.array([
        [cx, 0, 0],
        [0, cy, 0],
        [0, 0, 1]], dtype=np.float32)

def rotation_matrix(angle):
    return np.array([
        [cos(angle), sin(angle), 0],
        [-sin(angle), cos(angle), 0],
        [0, 0, 1]], dtype=np.float32)

def translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]], dtype=np.float32)

def horizontal_shear_matrix(sh):
    return np.array([
        [1, sh, 0],
        [0, 1, 0],
        [0, 0, 1]], dtype=np.float32)

def vertical_shear_matrix(sv):
    return np.array([
        [1, 0, 0],
        [sv, 1, 0],
        [0, 0, 1]], dtype=np.float32)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path, fov=60, baseline=0.3, width=1937):
    normalized_depth = depthmap_loader(path)
    focal = width / (2.0 * np.tan((fov * np.pi) / 360.0))
    ref_disp = (focal * baseline) / normalized_depth
    idx = ref_disp < 0.5
    ref_disp[idx] = 0
    idx = ref_disp > 191
    ref_disp[idx] = 191
    # ref_disp = np.expand_dims(ref_disp, axis=2)
    return ref_disp.astype('float32')

def depthmap_loader(path):
    depth = Image.open(path).convert("RGB")
    array = np.array(depth, dtype=np.float32)
    normalized_depth = np.dot(array[:, :, :], [1.0, 256.0, 65536.0])
    normalized_depth = normalized_depth / ((256.0 * 256.0 * 256.0) - 1)
    normalized_depth = normalized_depth * 1000

    return normalized_depth

class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, right_transforms, randomCrop, loader=default_loader, disploader=disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.disploader = disploader
        self.crop = randomCrop
        self.right_transforms = right_transforms


    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.disploader(disp_L)

        head, tail = os.path.split(self.left[index])

        frame_num = int(tail.split(".png")[0])

        transform = None

        for transl in self.right_transforms:
            if(transl.frame_num == frame_num):
                transform = transl

        print(transform.x)

        if ("_ch_x" in right):
            T = translation_matrix(transform.x, 0)

        elif ("_ch_y" in right):
            T = translation_matrix(0, transform.y)

        elif ("_ch_z" in right):
            T = scaling_matrix(1-(transform.z), 1-(transform.z))

        elif ("_ch_roll" in right):
            T = rotation_matrix(math.radians(transform.roll * -1.1))
            
        matrix = np.float32(T.flatten()[:6].reshape(2,3))
        width, height = right_img.size
        right_img = cv2.warpAffine(src=np.array(right_img), M=matrix, dsize=(width, height))

        processed = preprocess.get_transform(augment=False)

        if self.crop:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:
            w, h = left_img.size
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
