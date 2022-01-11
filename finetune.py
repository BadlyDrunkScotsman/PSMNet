from __future__ import print_function

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from subprocess import run
from torch.autograd import Variable

from dataloader import KITTILoader as DA

from models import *

from clearml import Task, Logger


from models.upscaling.upscaling_model import Generator
from PIL import Image

from utils.imgproc import *


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
        

task = Task.init("SR-PSMNET", "Kitty-2015-Scene_flow")
switch_to_poziomka(task)
ugly_hack()

maxdisp = 192
seed = 1
loadmodel = None
model_type = 'stackhourglass'
datatype = 'custom'
datapath = '/mnt/host/SSD/VIDAR/dane/hello_kitti_2015/data_scene_flow/training/'
savemodel = '/mnt/host/SSD/VIDAR/trash/sr_psmnet_results_x4_kitty/'
no_cuda = False
epochs = 30

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


TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True),
    batch_size=8, shuffle=True, num_workers=0, drop_last=False) # num_workers=0 this is very important 

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
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

if loadmodel is not None:
    state_dict = torch.load(loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true > 0)
    mask.detach_()
    # ----

    optimizer.zero_grad()

    if model_type == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
            output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask],
                                                                                  size_average=True)

    elif model_type == 'basic':
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3, 1)
        loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.item()


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    output3.squeeze_(dim=0)

    pred_disp = output3.data.cpu()

    # computing 3-px error#
    true_disp = copy.deepcopy(disp_true)
    index = np.argwhere(true_disp > 0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
                disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
            index[0][:], index[1][:], index[2][:]] * 0.05)
    torch.cuda.empty_cache()

    if(len(index[0]) == 0):
        return 0

    return 1 - (float(torch.sum(correct)) / float(len(index[0])))


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = 0.001
    else:
        lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



    
    








    max_acc = 0
    max_epo = 0
    start_full_time = time.time()

    for epoch in range(1, epochs + 1):
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer, epoch)

        # Training
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
            start_time = time.time()
            loss = train(imgL_crop, imgR_crop, disp_crop_L, model, cuda, model_type, optimizer)

            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            Logger.current_logger().report_scalar(
                "Training", "loss", iteration=((epoch - 1) * len(TrainImgLoader) + batch_idx), value=loss)

            total_train_loss += loss

        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
        Logger.current_logger().report_scalar(
                "Epoch_loss", "loss", iteration=epoch, value=(total_train_loss / len(TrainImgLoader)))

        # Test
        for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
            test_loss = test(imgL, imgR, disp_L, model, cuda)
            print('Iter %d 3-px error in val = %.3f' % (batch_idx, test_loss * 100))
            total_test_loss += test_loss


        print('epoch %d total 3-px error in val = %.3f' % (epoch, total_test_loss / len(TestImgLoader) * 100))
        Logger.current_logger().report_scalar(
                "Epoch_3-px_error", "3-px error", iteration=epoch, value=(total_test_loss / len(TestImgLoader) * 100))

        if total_test_loss / len(TestImgLoader) * 100 > max_acc:
            max_acc = total_test_loss / len(TestImgLoader) * 100
            max_epo = epoch
        print('MAX epoch %d total test error = %.3f' % (max_epo, max_acc))

        # SAVE
        savefilename = savemodel + 'finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
            'test_loss': total_test_loss / len(TestImgLoader) * 100,
        }, savefilename)

    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    print(max_epo)
    print(max_acc)
