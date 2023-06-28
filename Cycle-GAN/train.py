import argparse, os, glob
import torch,pdb
import math, random, time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model.model import _NetG,_NetD
#from dataset_dep import DatasetFromHdf5
from torchvision.utils import save_image
import torch.utils.model_zoo as model_zoo
from random import randint, seed
import random
import cv2
from dataloader.EyeQ import EyeQ_Dataset
import albumentations as A 
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import itertools

# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet") 
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=200, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=100, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument("--cuda", default=True, help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to resume model (default: none")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=0, help="Number of threads for data loader to use, (default: 1)")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model (default: none)")
parser.add_argument("--noise_sigma", default=50, type=int, help="standard deviation of the Gaussian noise (default: 50)")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--sigma", default=30, type=int)
parser.add_argument("--num_rand",default=[1000,1000], type=list)
parser.add_argument("--root", default="dataset/preprocessed/ALLDATA", type=str)
parser.add_argument("--file_dir",default="dataset/Label_EyeQ_train.csv", type=str)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num

def main():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    cuda = opt.cuda
    # if cuda: 
    #     print("=> use gpu id: '{}'".format(opt.gpus))
    #     os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    #     if not torch.cuda.is_available():
    #             raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    # data_list = glob.glob(opt.trainset+"*.h5")
    num_random = opt.num_rand
    print("===> Building model")
    g_AB = _NetG()
    g_BA = _NetG()
    discr_A = _NetD()
    discr_B = _NetD()

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    #网络参数数量
    # a,b=get_parameter_number(model)
    # print(model)
    # print(a,b)
    print("===> Setting GPU")
    if cuda:
        #model = model.cuda()
        #discr = discr.cuda()
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        #  dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            g_AB = nn.DataParallel(g_AB)
            g_BA = nn.DataParallel(g_BA)
            discr_A = nn.DataParallel(discr_A)
            discr_B = nn.DataParallel(discr_B)

        g_AB.to(device=device)
        g_BA.to(device=device)
        discr_A.to(device=device)
        discr_B.to(device=device)

        #discr.to(device=device)
        criterion_GAN = criterion_GAN.cuda()
        criterion_cycle = criterion_cycle.cuda()
        criterion_identity = criterion_identity.cuda()
        

    # optionally resume from a checkpoint
    # if opt.resume:
    #     if os.path.isfile(opt.resume):
    #         print("=> loading checkpoint '{}'".format(opt.resume))
    #         checkpoint = torch.load(opt.resume)
    #         opt.start_epoch = checkpoint["epoch"] + 1
    #         model.load_state_dict(checkpoint["model"].state_dict())
    #         discr.load_state_dict(checkpoint["discr"].state_dict())
    #     else:
    #         print("=> no checkpoint found at '{}'".format(opt.resume))

    # # optionally copy weights from a checkpoint
    # if opt.pretrained:
    #     if os.path.isfile(opt.pretrained):
    #         print("=> loading model '{}'".format(opt.pretrained))
    #         weights = torch.load(opt.pretrained)
    #         model.load_state_dict(weights['model'].state_dict())
    #         discr.load_state_dict(weights['discr'].state_dict())
    #     else:
    #         print("=> no model found at '{}'".format(opt.pretrained))

    data_transforms = {
        'HQ': T.Compose([
                T.Resize((512,512)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                #T.RandomRotation((-180,180)),
                T.ToTensor()
        ]),
        'LQ': T.Compose([
                T.Resize((512,512)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                #T.RandomRotation((-180,180)),
                T.ToTensor()
                #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("===> Setting Optimizer")
    optimizer_G = torch.optim.RMSprop(
        itertools.chain(g_BA.parameters(), g_AB.parameters()), lr=opt.lr/2
    )

    optimizer_D_A = torch.optim.RMSprop(discr_A.parameters(), lr=opt.lr)
    optimizer_D_B = torch.optim.RMSprop(discr_B.parameters(), lr=opt.lr)    
    #G_optimizer = optim.RMSprop(model.parameters(), lr=opt.lr/2)
    #D_optimizer = optim.RMSprop(discr.parameters(), lr=opt.lr)

    print("===> Training")
    all_g =[]
    all_d=[]
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        g_loss = 0
        d_loss = 0
        num_rand = 0
        for i in num_random:
            train_set = EyeQ_Dataset(root=opt.root,file_dir=opt.file_dir,select_number=i,transform_HQ=data_transforms['HQ'],transform_PQ=data_transforms['LQ'])
            #train_set = DatasetFromHdf5(data_name)
            training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, \
                batch_size=opt.batchSize, shuffle=True)
            a,b=train(training_data_loader,criterion_GAN,criterion_cycle,criterion_identity, optimizer_G, optimizer_D_A,optimizer_D_B, g_AB,g_BA, discr_A, discr_B, epoch,num_rand)
            g_loss += a
            d_loss += b
            num_rand += 1
        g_loss = g_loss / len(num_random)
        d_loss = d_loss / len(num_random)
        all_g.append(format(g_loss))
        all_d.append(format(d_loss))
        save_checkpoint(g_AB, discr_A, epoch)

        print(g_loss)

    file = open('cycleGan/Experiment/exp_512/checksample/mse_'+str(opt.nEpochs)+'_'+str(opt.sigma)+'.txt','w')
    for g in all_g:
        file.write(g+'\n')
    file.close()

    file = open('cycleGan/Experiment/exp_512/checksample/Gloss_'+str(opt.nEpochs)+'_'+str(opt.sigma)+'.txt', 'w')
    for d in all_d:
        file.write(d + '\n')
    file.close()
    # psnr = eval_dep(model)
    # print("Final psnr is:",psnr)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


def train(training_data_loader,criterion_GAN,criterion_cycle,criterion_identity,optimizer_G, optimizer_D_A,optimizer_D_B, g_AB,g_BA, discr_A, discr_B, epoch,num_rand):
    # training_data_loader, criterion_GAN,criterion_cycle,criterion_identity ,
    # optimizer_G, optimizer_D_A,optimizer_D_B, g_AB,g_BA, discr_A, discr_B, epoch,num_rand
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    lr = adjust_learning_rate(optimizer_D_A, epoch-1)

    all_loss_g = []
    all_loss_d = []
    for param_group in optimizer_G.param_groups:
        param_group["lr"] = lr/2
    for param_group in optimizer_D_B.param_groups:
        param_group["lr"] = lr
    for param_group in optimizer_D_A.param_groups:
        param_group["lr"] = lr

    print("Epoch={}, lr={}".format(epoch, optimizer_D_A.param_groups[0]["lr"]))
    #model.train()
    #discr.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        
        real_B = Variable(batch[1]) # B 
        real_A = Variable(batch[0])    # A
        #print(target)
        
        valid = Variable(torch.cuda.FloatTensor(np.ones((real_A.size(0), *discr_A.module.output_shape))), requires_grad=False)
        fake = Variable(torch.cuda.FloatTensor(np.zeros((real_A.size(0), *discr_B.module.output_shape))), requires_grad=False)

        if opt.cuda:
            real_A = real_A.cuda()
            real_B = real_B.cuda()

        # train Generator
        g_AB.train()
        g_BA.train()

        optimizer_G.zero_grad()

        #output = g_BA(real_A)
        #output_2 = g_AB(real_B)
        loss_id_A = criterion_identity(g_BA(real_A),real_A)
        loss_id_B = criterion_identity(g_AB(real_B),real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2
        
            # Gan loss
        fake_B = g_AB(real_A)
        loss_GAN_AB = criterion_GAN(discr_B(fake_B),valid)
        fake_A = g_BA(real_B)
        loss_GAN_BA = criterion_GAN(discr_A(fake_A),valid)


        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # cycle Loss
        
        recov_A = g_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A,real_A)
        recov_B = g_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B,real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Totall Loss 
        loss_G = loss_GAN + 10*loss_cycle + 5* loss_identity

        all_loss_g.append(loss_G)

        loss_G.backward()
        optimizer_G.step()


        # train discriminator A
        optimizer_D_A.zero_grad()
        # Real loss
        
        loss_real = criterion_GAN(discr_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(discr_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()


        # train discriminator B 
        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(discr_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(discr_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2        
        all_loss_d.append(loss_D)
        
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss_D: {:.5}, loss_G: {:.5}".format(epoch, iteration, len(training_data_loader), loss_D.data, loss_G.data))
    save_image(fake_B.data, 'cycleGan/Experiment/exp_512/checksample/'+str(epoch)+'_'+str(num_rand)+'_'+'output.png')
    save_image(real_A.data, 'cycleGan/Experiment/exp_512/checksample/'+str(epoch)+'_'+str(num_rand)+'_'+'input.png')
    save_image(real_B.data, 'cycleGan/Experiment/exp_512/checksample/'+str(epoch)+'_'+str(num_rand)+'_'+'gt.png')


    return torch.mean(torch.FloatTensor(all_loss_g)),torch.mean(torch.FloatTensor(all_loss_d))
   
def save_checkpoint(model, discr, epoch):
    model_out_path = "cycleGan/Experiment/exp_512/checkpoint/" + "model_denoise_"+str(epoch)+"_"+str(opt.sigma)+".pth"
    state = {"epoch": epoch ,"model": model, "discr": discr}
    if not os.path.exists("cycleGan/Experiment/exp_512/checkpoint/"):
        os.makedirs("cycleGan/Experiment/exp_512/checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()