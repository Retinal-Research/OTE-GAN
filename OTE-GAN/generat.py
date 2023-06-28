import argparse
import os, pdb
from turtle import shape
import torch, cv2
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import time, math, glob
import scipy.io as sio
from PIL import Image
from Helper.ssim import calculate_ssim_floder
from torchvision.utils import save_image
from dataloader.EyeQ_compare import EyeQ_Dataset
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from model.model import _NetG,_NetD


parser = argparse.ArgumentParser(description="PyTorch SRResNet Eval")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
#parser.add_argument("--model", default="SottGan/Experiment/exp1/checkpoint/model_denoise_198_40.pth", type=str, help="model path")
#parser.add_argument("--model", default="SottGan/Experiment/exp9/checkpoint/model_denoise_200_40.pth", type=str, help="model path")
parser.add_argument("--model", default="SottGan/Experiment/exp11/checkpoint/model_denoise_200_45.pth", type=str, help="model path")
# parser.add_argument("--save", default="SottGan/NR_result2", type=str, help="savepath, Default: results")
# parser.add_argument("--ori", default="SottGan/NR_result2", type=str, help="savepath, Default: results")
# parser.add_argument("--gpus", default="0", type=str, help="gpu ids")
# parser.add_argument("--root", default="dataset/preprocessed/ALLDATA", type=str)
# parser.add_argument("--file_dir",default="dataset/Label_EyeQ_test.csv", type=str)
parser.add_argument("--save", default="SottGan/NR_train", type=str, help="savepath, Default: results")
parser.add_argument("--ori", default="SottGan/NR_train", type=str, help="savepath, Default: results")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids")
parser.add_argument("--root", default="dataset/preprocessed/ALLDATA", type=str)
parser.add_argument("--file_dir",default="dataset/Label_EyeQ_train.csv", type=str)

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


opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
cuda = True#opt.cuda

if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

if not os.path.exists(opt.save):
    os.mkdir(opt.save)

#model = _NetG()
model = torch.load(opt.model)["model"]

train_set = EyeQ_Dataset(root=opt.root,file_dir=opt.file_dir,transform_HQ=data_transforms['HQ'],transform_PQ=data_transforms['LQ'])
training_data_loader = DataLoader(dataset=train_set,batch_size=1)
 
with torch.no_grad():
    for iteration, batch in enumerate(training_data_loader):
        #print(batch[2])
        im_input = batch[0]
        drlabel = batch[1]
        name  = batch[2]

        if cuda:
            model = model.cuda()
            im_input = im_input.cuda()
        else:
            model = model.cpu()

        im_output = model(im_input)

        im_output = im_output
        #im_output = torch.clamp(im_output,min=0.0,max=1.0)
        #print(im_output.shape)

        #result = torch.zeros(1,3,256,256*2)

        #result[0,:,:,:256] = im_input
        #result[0,:,:,256:] = im_output

       # save_image(result.cpu().data,opt.ori+'/'+name[-1])

        save_image(im_output.cpu().data,opt.save+'/'+name[-1])
        #save_image(im_input.cpu().data,opt.ori+'/'+name[-1])
