import random

import numpy as np
import torch.utils.data as data
from PIL import Image
import cv2
import os

from utils import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def translate(img, x, y, z):
    # get the width and height of the image
    height, width = img.shape[:2]

    # get tx and ty values for translation
    # you can specify any value of your choice
    tx, ty = 1 - ((width*x)/10), 1 - ((height*y)/10)

    # create the translation matrix using tx and ty, it is a NumPy array
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty]], dtype=np.float32)
    
    return cv2.warpAffine(src=img, M=translation_matrix, dsize=(width, height))

def disparity_loader(path, fov=60, baseline=0.4, width=1937):
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
    def __init__(self, left, right, left_disparity, randomCrop, trans_data_file_path, loader=default_loader, disploader=disparity_loader):
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.disploader = disploader
        self.crop = randomCrop
        self.trans_data_file_path = trans_data_file_path

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.disploader(disp_L)

        # removing extension
        file_name = os.path.splitext(os.path.basename(self.trans_data_file_path))[0]

        # Using readlines()
        file = open(self.trans_data_file_path, 'r')
        Lines = file.readlines()

        
        trans_x = 0
        trans_y = 0
        trans_z = 0
        
        # Strips the newline character
        for line in Lines:
            line = line.strip()
            values = line.split(' ')
            if(values[0] == file_name):
                trans_x = values[1]
                trans_y = values[2]
                trans_z = values[3]

        file.close()

        right_img = translate(right_img, trans_x, trans_y, trans_z)

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
