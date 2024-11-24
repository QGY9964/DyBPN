import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np



import torch
from torch.utils.data import DataLoader


def get_dataloader(datapath, scaler_X, scaler_Y, batch_size, mode='train'):
    datapath = os.path.join(datapath, mode)
    # cuda = True if torch.cuda.is_available() else False
    cuda = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    X = Tensor(np.expand_dims(np.load(os.path.join(datapath, 'X.npy')), 1)) / scaler_X
    Y = Tensor(np.expand_dims(np.load(os.path.join(datapath, 'Y.npy')), 1)) / scaler_Y
    ext = Tensor(np.load(os.path.join(datapath, 'ext.npy')))
    assert len(X) == len(Y)
    print('# {} samples: {}'.format(mode, len(X)))

    data = torch.utils.data.TensorDataset(X, ext, Y)
    if mode == 'train':
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    else:
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    return dataloader


def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {:2f}M'.format(str, total_num/(1024*1024)))
