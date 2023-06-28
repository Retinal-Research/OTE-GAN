import argparse
import os, pdb
import torch, cv2
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import time, math, glob
import scipy.io as sio
from PIL import Image
from Helper.ssim import calculate_ssim_floder
from torchvision.utils import save_image
from dataloader.EyeQ_Test import EyeQ_Dataset
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model.model import _NetG,_NetD


parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="SottGan/Experiment/exp1/checkpoint/model_denoise_198_40.pth", type=str, help="model path")
#parser.add_argument("--model", default="SottGan/Experiment/exp8/checkpoint/model_denoise_190_40.pth", type=str, help="model path")
parser.add_argument("--save", default="SottGan/FR_result", type=str, help="savepath, Default: results")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")
parser.add_argument("--dataset", default='dataset/preprocessed/referencetest/pre',type=str)
parser.add_argument("--input",default='dataset/preprocessed/referencetest/deg',type=str)

def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt((imdff ** 2).mean())
    if rmse == 0:
        return 100  
    return 20 * math.log10(1.0 / rmse)


data_transforms = {
        'HQ': T.Compose([
                T.Resize((256,256)),
                #T.RandomRotation((-180,180)),
                T.ToTensor()
        ]),
        'LQ': T.Compose([
                T.Resize((256,256)),
                #T.RandomRotation((-180,180)),
                T.ToTensor()
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

train_set = EyeQ_Dataset(mode='111',transform_HQ=data_transforms['HQ'],transform_PQ=data_transforms['LQ'])
training_data_loader = DataLoader(dataset=train_set,batch_size=1)


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if not os.path.exists(opt.save):
    os.mkdir(opt.save)

#model = _NetG()
model = torch.load(opt.model)["model"]


 
p=0
p2=0
with torch.no_grad():
    for iteration, batch in enumerate(training_data_loader):

        im_input = batch[0]
        im_gt = batch[1]
        name  = batch[2]
        if cuda:
            model = model.cuda()
            im_gt=im_gt.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()

        start_time = time.time()

        height = int(im_input.size()[2])
        width = int(im_input.size()[3])
        M = int(height / 16)  # 行能分成几组
        N = int(width / 16)

        im_input = im_input[:,:, :M * 16, :N * 16]
        im_gt = im_gt[:,:, :M * 16, :N * 16]
        im_output = torch.zeros(3, M * 16, N * 16)


        im_output = model(im_input)
        im_output = torch.clamp(im_output,min=0.0,max=1.0)
        pp=PSNR(im_output,im_gt)
        pp2=PSNR(im_input,im_gt)
        p+=pp
        p2+=pp2
        #HR_4x = HR[:,:,:,:,0].cpu()
        im_output = im_output.cpu()
        # save_image(im_output.data,'6.png')
        save_image(im_output.data,opt.save+'/'+name[-1])

ssim=calculate_ssim_floder(opt.dataset,opt.save)
ssim_input=calculate_ssim_floder(opt.dataset,opt.input,mode='input')
print("Average PSNR:",p/len(train_set))
print("Average input PSNR:",p2/len(train_set))
print("Average SSIM:",ssim)
print("Average Input SSIM:",ssim_input)