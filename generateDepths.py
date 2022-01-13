from __future__ import print_function

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from subprocess import run
from torch.autograd import Variable
import numpy as np

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


task = Task.init("PSMNET_Translacja", "Predict")
switch_to_poziomka(task)
ugly_hack()

maxdisp = 192
seed = 1
loadmodel = None
model_type = 'stackhourglass'
datatype = 'custom'

datapath = '/mnt/host/SSD/VIDAR/dane/20220107-2226_3_60FOV_BL30_NOWY/'

model_path = '/mnt/host/SSD/VIDAR/trash/psmnet_aug_translaction/checkpoint_b30_fov60_29.tar'
outpath = '/mnt/host/SSD/VIDAR/trash/uncalibrated_psmnet_results/'
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

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp, right_transforms = ls.dataloader(datapath)

from dataloader import CustomLoader as DA

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, right_transforms, False),
    batch_size=1, shuffle=False, num_workers=0, drop_last=False)

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
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


def test(imgL, imgR, disp_true):
    model.eval()

    if cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
    # ---------
    mask = disp_true < 192
    # ----

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2] // 16
        top_pad = (times + 1) * 16 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3] // 16
        right_pad = (times + 1) * 16 - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3, 0)

    #print(output3.shape)

    if top_pad != 0:
        img = output3[:, top_pad:, right_pad:]
    else:
        img = output3

    im_m = img[mask]
    d_t = disp_true[mask]

    loss = F.l1_loss(im_m, d_t)  # torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

    return loss.data.cpu(), img


start_full_time = time.time()
for batch_idx, (imgL, imgR, disp_L, r_path) in enumerate(TestImgLoader):
    test_loss, result = test(imgL, imgR, disp_L)
    print('Iter %d loss = %.3f' % (batch_idx, test_loss))

    result = result.numpy()
    disp_L = disp_L.numpy()

    img = (result * 256).astype('uint16')
    img = Image.fromarray(img)

    gt_disp = np.asarray(disp_L * 256).astype('uint16')
    gt_disp = Image.fromarray(gt_disp)

    head, tail = os.path.split(r_path)
    img_name = tail

    dirname = os.path.basename(
        os.path.dirname(r_path))

    
    curr_outpath = outpath + dirname + "/"

    curr_gt_outpath = outpath + dirname + "_gt/"

    os.mkdir(curr_outpath)
    os.mkdir(curr_gt_outpath)

    img.save(os.path.join(curr_outpath, img_name))
    gt_disp.save(os.path.join(curr_gt_outpath, img_name))
    
print('full predicting time = %.2f HR' % ((time.time() - start_full_time) / 3600))

