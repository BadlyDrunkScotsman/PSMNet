from __future__ import print_function

import argparse
import time
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from subprocess import run
from torch.autograd import Variable


from dataloader import CustomLoader as DA
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


def main():
    task = Task.init("PSMNET", "multi Depth-map generation")
    switch_to_poziomka(task)
    ugly_hack()

    import cv2

    maxdisp = 192
    seed = 1
    model_path = '/mnt/host/SSD/VIDAR/trash/pretrained_model_KITTI2015.tar'
    model_type = 'stackhourglass'
    datatype = '2015'
    datapath = '/mnt/host/SSD/VIDAR/dane/kitti/data_scene_flow/training/'
    
    outpath = '/mnt/host/SSD/VIDAR/trash/psmnet_kitti_results'

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
        from dataloader import CustomDataSetLoader as ls

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
        
            imgL_o = Image.open(test_left_img[idx]).convert('RGB')
            imgR_o = Image.open(test_right_img[idx]).convert('RGB')

            disp_L = cv2.imread(test_left_disp[idx], cv2.IMREAD_UNCHANGED)

            w, h = imgL_o.size

            start_time = time.time()
            
            imgL = infer_transform(imgL_o)
            imgR = infer_transform(imgR_o)

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

            gt_disp = numpy.asarray(disp_L).astype('uint16')
            gt_disp = Image.fromarray(gt_disp)
            
            head, tail = os.path.split(test_left_disp[idx])

            img_name = tail

            print("procesing: " + os.path.join(predicted_dir, img_name))

            img.save(os.path.join(predicted_dir, img_name))
            gt_disp.save(os.path.join(ground_truth_dir, img_name))

if __name__ == '__main__':
    main()
