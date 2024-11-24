import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import warnings
import numpy as np
import argparse
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utils.metrics import get_MSE, get_MAE, get_MAPE
from utils.data_process import get_dataloader, print_model_parm_nums
# from pdd_my_model_base import Model, weights_init_normal

# from tttt import Model
# from pdd_my_model_base import Model_F

from DyBPN import DyBPN

# load arguments
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=1e-4, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
parser.add_argument('--base_channels', type=int, default=32, help='number of feature maps')
parser.add_argument('--img_width', type=int, default=32, help='image width')
parser.add_argument('--img_height', type=int, default=32, help='image height')
parser.add_argument('--in_ch', type=int, default=1, help='number of flow image channels')
parser.add_argument('--out_ch', type=int, default=1, help='number of flow image channels')
parser.add_argument('--scaler_X', type=int, default=1500, help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=100, help='scaler of fine-grained flows')
parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
parser.add_argument('--seed', type=int, default=2017, help='random seed')
parser.add_argument('--harved_epoch', type=int, default=10, help='halved at every x interval')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--dataset', type=str, default='P1', help='which dataset to use')
parser.add_argument('--model', type=str, default='DyBPN', help=' which model')
opt = parser.parse_args()
print(opt)
torch.manual_seed(opt.seed)
warnings.filterwarnings('ignore')
# path for saving model

save_path = 'saved_model/{}/{}/{}/{}'.format(opt.dataset_type, opt.dataset, opt.fraction, opt.model)
os.makedirs(save_path, exist_ok=True)

# Train CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# initial model
model = DyBPN(in_channels=opt.in_ch,
              base_channels=opt.base_channels,
              scaler_X=opt.scaler_X,
              scaler_Y=opt.scaler_Y,
              img_width=opt.img_width,
              img_height=opt.img_height,
              out_channels=opt.out_ch
              )
torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=5)
print_model_parm_nums(model, 'Model')

criterion = nn.L1Loss()

if cuda:
    model.cuda()
    criterion.cuda()

# load training set and validation set


datapath = os.path.join('./data', opt.dataset)
train_dataloader = get_dataloader(datapath, opt.scaler_X, opt.scaler_Y, opt.batch_size, 'train')
valid_dataloader = get_dataloader(datapath, opt.scaler_X, opt.scaler_Y, 4, 'valid')

val_sample_interval = 0
if opt.dataset == "P1":
    val_sample_interval = 96 * opt.fraction / 100
elif opt.dataset == "P2":
    val_sample_interval = 112 * opt.fraction / 100
elif opt.dataset == "P3":
    val_sample_interval = 100 * opt.fraction / 100
elif opt.dataset == "P4":
    val_sample_interval = 133 * opt.fraction / 100
# Optimizers
lr = opt.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.b1, opt.b2))
# training phase
iter = 0
rmses = [np.inf]
maes = [np.inf]


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr[-1]


for epoch in range(opt.n_epochs):
    train_loss = 0

    ep_time = datetime.now()
    for i, (flows_c, ext, flows_f) in enumerate(train_dataloader):
        model.train()
        optimizer.zero_grad()

        gen_hr = model(flows_c, ext)

        loss = criterion(gen_hr, flows_f)
        loss.backward()
        optimizer.step()

        if iter % val_sample_interval == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch,
                                                                    opt.n_epochs,
                                                                    i,
                                                                    len(train_dataloader),
                                                                    np.sqrt(loss.item())))

        # counting training mse
        # validation phase
        if iter % val_sample_interval == 0:
            model.eval()
            valid_time = datetime.now()
            total_mse, total_mae, total_mape = 0, 0, 0
            for j, (flows_c, ext, flows_f) in enumerate(valid_dataloader):
                preds = model(flows_c, ext)

                preds = preds.cpu().detach().numpy() * opt.scaler_Y
                flows_f = flows_f.cpu().detach().numpy() * opt.scaler_Y
                total_mse += get_MSE(preds, flows_f) * len(flows_c)
                total_mae += get_MAE(preds, flows_f) * len(flows_c)

                total_mape += get_MAPE(preds, flows_f) * len(flows_c)

            rmse = np.sqrt(total_mse / len(valid_dataloader.dataset))
            mae = np.sqrt(total_mae / len(valid_dataloader.dataset))
            mape = np.sqrt(total_mape / len(valid_dataloader.dataset))

            print("iter\t{}\tNow_Validaton_RMSE\t{:.6f}\ttime\t{}".format(iter,
                                                                          rmse, datetime.now() - valid_time))
            f = open('{}/results2.txt'.format(save_path), 'a')
            f.write(
                "epoch\t{}\titer\t{}\tRMSE\t{:.6f}\tMAE\t{:.6f}\tMAPE\t{:.6f}\n".format(epoch, iter, rmse, mae, mape))

            if rmse < np.min(rmses):
                print("iter\t{}\tMin_RMSE\t{:.6f}\ttime\t{}".format(iter, rmse, datetime.now() - valid_time))
                torch.save(model.state_dict(),
                           '{}/final_model.pt'.format(save_path))
                f = open('{}/results.txt'.format(save_path), 'a')
                f.write("epoch\t{}\titer\t{}\tRMSE\t{:.6f}\n".format(epoch, iter, rmse))
                f.close()
            rmses.append(rmse)

        iter += 1
    # halve the learning rate

    if epoch % opt.harved_epoch == 0 and epoch != 0:
        lr /= 5
        print('lr=', lr)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=(opt.b1, opt.b2))
        f = open('{}/results.txt'.format(save_path), 'a')
        f.write("de the learning rate!\n")
        f.close()

    print('=================time cost: {}==================='.format(
        datetime.now() - ep_time))
