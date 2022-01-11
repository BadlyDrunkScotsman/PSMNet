from __future__ import print_function

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from subprocess import run
from torch.autograd import Variable

from models import *

from clearml import Task, Logger
from PIL import Image

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


task = Task.init("PSMNET_Translacja", "Main")
switch_to_poziomka(task)
ugly_hack()

maxdisp = 192
seed = 1
loadmodel = None
model_type = 'stackhourglass'
datatype = 'custom'
datapath = '/mnt/host/SSD/VIDAR/dane/20220107-2226_3_60FOV_BL30_NOWY/'
savemodel = '/mnt/host/SSD/VIDAR/trash/psmnet_aug_translaction/'
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

from dataloader import CustomLoader as DA

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_left_disp, right_transforms, True),
    batch_size=8, shuffle=True, num_workers=0, drop_last=False) # num_workers=0 this is very important 

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

if loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


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
        output3 = torch.squeeze(output3)

    if top_pad != 0:
        img = output3[:, top_pad:, :]
    else:
        img = output3

    if len(disp_true[mask]) == 0:
        loss = 0
    else:
        loss = F.l1_loss(img[mask],
                         disp_true[mask])  # torch.mean(torch.abs(img[mask]-disp_true[mask]))  # end-point-error

    return loss.data.cpu()


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


start_full_time = time.time()
for epoch in range(0, epochs):
    print('This is %d-th epoch' % (epoch))
    total_train_loss = 0
    adjust_learning_rate(optimizer, epoch)

    ## training ##
    for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
        start_time = time.time()

        loss = train(imgL_crop, imgR_crop, disp_crop_L, model, cuda, model_type, optimizer)
        print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
        total_train_loss += loss

        Logger.current_logger().report_scalar(
            "Training", "loss", iteration=(epoch * len(TrainImgLoader) + batch_idx), value=loss)
        
    print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))
    Logger.current_logger().report_scalar(
            "Epoch_loss", "loss", iteration=epoch, value=(total_train_loss / len(TrainImgLoader)))

    total_test_loss = 0
    for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
        test_loss = test(imgL, imgR, disp_L, model, cuda)
        print('Iter %d test loss = %.3f' % (batch_idx, test_loss))
        total_test_loss += test_loss
    print('total test loss = %.3f' % (total_test_loss / len(TestImgLoader)))
    Logger.current_logger().report_scalar(
                "Epoch_l1_loss", "l1_loss", iteration=epoch, value=(total_test_loss / len(TestImgLoader)))
        

    # SAVE
    savefilename = savemodel + '/checkpoint_b30_fov60_' + str(epoch) + '.tar'
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'train_loss': total_train_loss / len(TrainImgLoader),
    }, savefilename)

print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))

