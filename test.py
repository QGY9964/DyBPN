import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import warnings
import argparse
from argparse import Namespace

import numpy as np
import torch
from utils.data_process import get_dataloader, print_model_parm_nums
from utils.metrics import get_MSE, get_MAE, get_MAPE

from DyBPN import DyBPN

warnings.filterwarnings("ignore")
# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--base_channels', type=int, default=32, help='number of feature maps')
parser.add_argument('--img_width', type=int, default=32, help='image width')
parser.add_argument('--img_height', type=int, default=32, help='image height')
parser.add_argument('--in_ch', type=int, default=1, help='number of flow image channels')
parser.add_argument('--out_ch', type=int, default=1, help='number of flow image channels')
parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
parser.add_argument('--scaler_X', type=int, default=1500, help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=100, help='scaler of fine-grained flows')
parser.add_argument('--ext_dim', type=int, default=7, help='external factor dimension')
parser.add_argument('--ext_flag', action='store_true', help='whether to use external factors')
parser.add_argument('--use_exf', action='store_true', default=True, help='External influence factors')
parser.add_argument('--dataset', type=str, default='P1', choices=["P1", "P2", "P3", "P4"], help='which dataset to use')
parser.add_argument('--dataset_type', type=str, default='bj', help='which dataset type to use')
parser.add_argument('--model', type=str, default='DyBPN', help=' which model')
opt: Namespace = parser.parse_args()

print(opt)
model_path = 'saved_model/{}/{}'.format(opt.dataset, opt.model)
print(model_path)
# test CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# load model

model = DyBPN(in_channels=opt.in_ch,
              base_channels=opt.base_channels,
              scaler_X=opt.scaler_X,
              scaler_Y=opt.scaler_Y,
              img_width=opt.img_width,
              img_height=opt.img_height,
              out_channels=opt.out_ch
              )

print(opt.model)
model.load_state_dict(torch.load('{}/final_model.pt'.format(model_path)))
model.eval()
if cuda:
    model.cuda()
print_model_parm_nums(model, 'Model')
# load test set
datapath = os.path.join('../data', opt.dataset_type, opt.dataset)
dataloader = get_dataloader(datapath, opt.scaler_X, opt.scaler_Y, 4, 'test')
total_mse, total_mae, total_mape = 0, 0, 0


def l1(labels, pred):
    out = np.abs(labels - pred)
    return out


def MSE(pred, real):
    mse = get_MSE(pred, real)
    print('Test: MSE={:.6f}'.format(mse))


keys = []
for i, (test_data, ext, test_labels) in enumerate(dataloader):
    print(i * 4)

    preds = model(test_data, ext).cpu().detach().numpy() * opt.scaler_Y
    test_labels = test_labels.cpu().detach().numpy() * opt.scaler_Y
    mse = get_MSE(preds, test_labels) * len(test_data)
    keys.append(mse)
    total_mse += get_MSE(preds, test_labels) * len(test_data)
    total_mae += get_MAE(preds, test_labels) * len(test_data)
    total_mape += get_MAPE(preds, test_labels) * len(test_data)

rmse = np.sqrt(total_mse / len(dataloader.dataset))
mae = total_mae / len(dataloader.dataset)
mape = total_mape / len(dataloader.dataset)
print('Test MSE = {:.3f}, MAE = {:.3f}, MAPE = {:.3f}'.format(rmse * rmse, mae, mape))
