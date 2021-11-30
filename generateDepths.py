from __future__ import print_function

import argparse
import time
import numpy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from subprocess import run
from torch.autograd import Variable

from models import *

from clearml import Task, Logger

import torchvision.transforms as transforms
from PIL import Image

import os
import os.path


def ugly_hack():
    oh = run(["apt-get", "update"])
    no = run("apt-get install ffmpeg libsm6 libxext6 -y".split(" "))

def switch_to_poziomka(task):
    if task.running_locally():
        task._wait_for_repo_detection()
        config = task.export_task()
        config['script']['repository'] = config['script']['repository'].replace("gitlab.com:", "gitlab.com-vidar:")
        task.update_task(config)
        task.execute_remotely(queue_name="default")


def test(imgL, imgR, model, cuda):
    model.eval()

    if cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()

    with torch.no_grad():
        disp = model(imgL, imgR)

    disp = torch.squeeze(disp)
    pred_disp = disp.data.cpu().numpy()

    return pred_disp



def feature_extraction(img, sift):
    # find the keypoints and descriptors with SIFT
    return sift.detectAndCompute(img, None)


def calcuate_homograpy_matrices(img_path1, img_path2, cv2, flann, sift):
        img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)

        kp1, des1 = feature_extraction(img1, sift)
        kp2, des2 = feature_extraction(img2, sift)

        matches = flann.knnMatch(des1, des2, k=2)

        # Keep good matches: calculate distinctive image features
        # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
        # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
        matchesMask = [[0, 0] for i in range(len(matches))]
        good = []
        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                # Keep this keypoint pair
                matchesMask[i] = [1, 0]
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        # ------------------------------------------------------------
        # STEREO RECTIFICATION

        # Calculate the fundamental matrix for the cameras
        # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

        # We select only inlier points
        pts1 = pts1[inliers.ravel() == 1]
        pts2 = pts2[inliers.ravel() == 1]

        # Stereo rectification (uncalibrated variant)
        # Adapted from: https://stackoverflow.com/a/62607343
        h1, w1 = img1.shape
        _, H1, H2 = cv2.stereoRectifyUncalibrated(
            np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
        )

        return H1 , H2

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

def main():
    task = Task.init("PSMNET", "multi Depth-map generation")
    switch_to_poziomka(task)

    ugly_hack()
    import cv2



    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary

    # Initiate flann
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    maxdisp = 192
    seed = 1
    model_path = '/mnt/host/SSD/VIDAR/trash/fov60_bs30_29.tar'
    model_type = 'stackhourglass'
    datatype = 'custom'
    datapath = '/mnt/host/SSD/VIDAR/dane/calib/20210901-1211_x/'
    
    outpath = '/mnt/host/SSD/VIDAR/trash/20210901-1211_x/'

    no_cuda = False

    cuda = not no_cuda and torch.cuda.is_available()

    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**normal_mean_var)])

    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    if datatype == '2015':
        from dataloader import KITTIloader2015 as ls
    elif datatype == '2012':
        from dataloader import KITTIloader2012 as ls
    elif datatype == 'custom':
        from dataloader import FullDataSetLoader as ls

    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(datapath)
    
    if model_type == 'stackhourglass':
        model = stackhourglass(maxdisp)
    elif model_type == 'basic':
        model = basic(maxdisp)
    else:
        print('no model')

    if cuda:
        model = nn.DataParallel(model)
        model.cuda()

    print('Load pretrained model')
    pretrain_dict = torch.load(model_path)
    model.load_state_dict(pretrain_dict['state_dict'])

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    ground_truth_dir = outpath + '/ground/'
    predicted_dir = outpath + '/predicted/'

    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)

    if not os.path.exists(predicted_dir):
        os.makedirs(predicted_dir)

    for idx in range(len(test_left_img)): #, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        
            imgL_o = cv2.imread(test_left_img[idx])
            imgR_o = cv2.imread(test_right_img[idx])
            
            h, w = imgL_o.shape[:2]

            H1, H2 = calcuate_homograpy_matrices(test_left_img[idx], test_right_img[idx], cv2, flann, sift)

            # Undistort (rectify) the images and save them
            # Adapted from: https://stackoverflow.com/a/62607343
            img1_rectified = cv2.warpPerspective(imgL_o, H1, (w, h))
            img2_rectified = cv2.warpPerspective(imgR_o, H2, (w, h))

            start_time = time.time()
            
            imgL = infer_transform(img1_rectified)
            imgR = infer_transform(img2_rectified)

            print(imgL.size())

            print('time = %.2f' % (time.time() - start_time))

            # pad to width and hight to 32 times
            if imgL.shape[1] % 32 != 0:
                times = imgL.shape[1] // 32
                top_pad = (times + 1) * 32 - imgL.shape[1]
            else:
                top_pad = 0

            if imgL.shape[2] % 32 != 0:
                times = imgL.shape[2] // 32
                right_pad = (times + 1) * 32 - imgL.shape[2]
            else:
                right_pad = 0

            imgL = F.pad(imgL, (0, right_pad, top_pad, 0)).unsqueeze(0)
            imgR = F.pad(imgR, (0, right_pad, top_pad, 0)).unsqueeze(0)

            print(imgL.size())

            pred_disp = test(imgL, imgR, model, cuda)

            print(pred_disp.shape)

            img = pred_disp[pred_disp.shape[0] - h:, :-(pred_disp.shape[1] - w)]
            
            img = (img * 256).astype('uint16')
            img = Image.fromarray(img)

            gt_disp = numpy.asarray(disparity_loader(test_left_disp[idx]) * 256).astype('uint16')
            gt_disp = Image.fromarray(gt_disp)
            
            head, tail = os.path.split(test_left_disp[idx])

            img_name = tail

            print("procesing: " + os.path.join(predicted_dir, img_name))

            img.save(os.path.join(predicted_dir, img_name))
            gt_disp.save(os.path.join(ground_truth_dir, img_name))

if __name__ == '__main__':
    main()
