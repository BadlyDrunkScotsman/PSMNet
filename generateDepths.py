from __future__ import print_function

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from subprocess import run
from torch.autograd import Variable


from dataloader import CustomLoader as DA
from models import *

from clearml import Task, Logger

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
    task = Task.init("SR-PSMNET", "multi Depth-map generator")
    switch_to_poziomka(task)
    ugly_hack()

    maxdisp = 192
    seed = 1
    model_path = '/mnt/host/SSD/VIDAR/modele/PSMnet/fov60_bs15_29.tar'
    model_type = 'stackhourglass'
    datatype = 'custom'
    datapath = '/mnt/host/SSD/VIDAR/dane/croped_ds_15b_fov60'
    
    outpath = '/mnt/host/SSD/VIDAR/trash/sr_psmnet_results'

    no_cuda = False

    cuda = not no_cuda and torch.cuda.is_available()

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

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=1, shuffle=False, num_workers=0, drop_last=False) # num_workers=0 this is very important 

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

    valid_file = open(os.path.join(datapath, 'valid.txt'), 'r')
    valid_lines = valid_file.readlines()

    ground_truth_dir = outpath + '/ground/'
    predicted_dir = outpath + '/predicted/'

    if not os.path.exists(ground_truth_dir):
        os.makedirs(ground_truth_dir)

    if not os.path.exists(predicted_dir):
        os.makedirs(predicted_dir)

    start_full_time = time.time()
    for idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            pred_disp = test(imgL, imgR, model, cuda)
            pred_disp = (pred_disp * 256).astype('uint16')
            pred_disp = Image.fromarray(pred_disp)

            gt_disp = torch.squeeze(disp_L)
            gt_disp = gt_disp.data.cpu().numpy()
            gt_disp = (gt_disp * 256).astype('uint16')
            gt_disp = Image.fromarray(gt_disp)
            
            img_name = valid_lines[idx].strip()

            print("procesing: " + os.path.join(predicted_dir, img_name))

            pred_disp.save(os.path.join(predicted_dir, img_name))
            gt_disp.save(os.path.join(ground_truth_dir, img_name))

    valid_file.close()

    print('full generating time = %.2f HR' % ((time.time() - start_full_time) / 3600))

if __name__ == '__main__':
    main()
