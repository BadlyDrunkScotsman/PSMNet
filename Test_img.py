from __future__ import print_function

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from subprocess import run
from PIL import Image

from clearml import Task, Logger

from models import *

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/
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
        task = Task.init("PSMNET", "Test")
        switch_to_poziomka(task)
        ugly_hack()

        cuda = True
        model = 'stackhourglass'
        maxdisp = 192


        leftimg = '/mnt/host/SSD/VIDAR/dane/croped_ds_15b_fov60/cam_60/44677.png' # path to left image
        rightimg = '/mnt/host/SSD/VIDAR/dane/croped_ds_15b_fov60/cam_60_15bs/44677.png' # path to right image

        resultimg = '/mnt/host/SSD/VIDAR/trash/44677.png' # path to out image

        model_path = '/mnt/host/SSD/VIDAR/modele/PSMNET/fov60_bs15_29.tar'

        seed = 1
        
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)
        
        model = stackhourglass(maxdisp)
        if cuda:
            model = nn.DataParallel(model, device_ids=[0])
            model.cuda()

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict['state_dict'])

        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])    

        imgL_o = Image.open(leftimg).convert('RGB')
        imgR_o = Image.open(rightimg).convert('RGB')

        imgL = infer_transform(imgL_o)
        imgR = infer_transform(imgR_o) 
       

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)

        start_time = time.time()
        pred_disp = test(imgL, imgR, model, cuda)
        print('time = %.2f' %(time.time() - start_time))

        
        if top_pad !=0 and right_pad != 0:
            img = pred_disp[top_pad:,:-right_pad]
        elif top_pad ==0 and right_pad != 0:
            img = pred_disp[:,:-right_pad]
        elif top_pad !=0 and right_pad == 0:
            img = pred_disp[top_pad:,:]
        else:
            img = pred_disp
        
        img = (img*256).astype('uint16')
        img = Image.fromarray(img)
        img.save(resultimg)

if __name__ == '__main__':
   main()






