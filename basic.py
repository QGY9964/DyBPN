import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DAF(nn.Module):
    def __init__(self, h=32, w=32, layer_scale_init_value=1e-6):
        super(DAF, self).__init__()
        self.h = h
        self.w = w
        self.embed_day = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
        self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
        self.embed_weather = nn.Embedding(18, 3)  # ignore 0, thus use 18
        self.img_width = h
        self.img_height = w
        self.ext2lr = nn.Sequential(
            nn.Linear(12, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(128, h * w),
            nn.ReLU(inplace=True)
        )

        self.conv_1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))
        self.layer_scale_init_value = layer_scale_init_value
        self.lamda1 = nn.Parameter(layer_scale_init_value * torch.ones(1),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, cg, ext):
        inp = cg
        ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
        ext_out2 = self.embed_hour(ext[:, 5].long().view(-1, 1)).view(-1, 3)
        ext_out3 = self.embed_weather(ext[:, 6].long().view(-1, 1)).view(-1, 3)
        ext_out4 = ext[:, :4]
        ext_lr1 = self.ext2lr(torch.cat([ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, self.img_width,
                                                                                               self.img_height)
        bs = cg.shape[0]
        cg = self.conv_1(cg)
        cg1 = cg.reshape(bs, -1, self.img_width * self.img_height)
        ext1 = ext_lr1.reshape(bs, self.img_width, self.img_height)

        inp = torch.cat((inp, ext_lr1), dim=1)
        inp1 = self.conv_2(inp)
        daf = torch.add((self.layer_scale_init_value *
                         self.lamda1 * torch.bmm(ext1, cg1)).reshape(bs, -1, self.h, self.w), inp1)

        return daf


###################################################################################
class GFN(nn.Module):
    def __init__(self, dim, thed=0.8):
        super(GFN, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.sig = nn.Sigmoid()
        self.thed = thed

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.sig(self.conv1(x))
        th = self.thed * (1 - torch.min(x2) / torch.max(x2))
        x3 = torch.where(x2 > th, torch.tensor(1), torch.tensor(0))

        out = x1 + torch.mul(x3, x1)
        out = self.conv1(out)
        return out


def attention(Q, K, V, dropout=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    att_scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        att_scores = dropout(att_scores)
    att_out = torch.matmul(att_scores, V)

    return att_out


###################################################################################
class SelfAttention(nn.Module):
    def __init__(self, d_model, hideen, head, img_h=32, img_w=32):
        super().__init__()

        self.norm = nn.InstanceNorm2d(d_model)

        self.hideen = hideen
        self.d_model = d_model
        self.head = head
        self.img_h = img_h
        self.img_w = img_w
        assert (d_model % head == 0)
        self.head = head
        self.d_model = d_model
        self.d_k = d_model // head

        self.Q = torch.nn.Conv2d(d_model, hideen, 1, 1, 0, bias=True)
        self.K = torch.nn.Conv2d(d_model, hideen, 1, 1, 0, bias=True)
        self.V = torch.nn.Conv2d(d_model, hideen, 1, 1, 0, bias=True)

        self.attn_out = nn.Conv2d(hideen, hideen, 1)

    def forward(self, x):
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        attn_out = attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        attn_out = attn_out.transpose(-2, -1).contiguous().view(batch_size, -1, self.img_h,
                                                                self.img_w)
        attn_out = self.attn_out(attn_out)

        attn_out = self.norm(attn_out)
        out_put = self.attn_out(attn_out)
        return out_put


###################################################################################
class MCA(nn.Module):
    def __init__(self, d_model, head=4, img_h=32, img_w=32):
        super(MCA, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        assert (d_model % head == 0)
        self.head = head
        self.d_model = d_model
        self.d_k = d_model // head

        self.Q = torch.nn.Conv2d(d_model, d_model, 1, 1, 0, bias=True)
        self.K = torch.nn.Conv2d(d_model, d_model, 1, 1, 0, bias=True)
        self.V = torch.nn.Conv2d(d_model, d_model, 1, 1, 0, bias=True)

        self.norm = nn.InstanceNorm2d(d_model)

        self.attn_out = GFN(d_model, d_model)

    def forward(self, x, y):
        q = self.Q(x)
        k = self.K(y)
        v = self.V(y)
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        attn_out = attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        attn_out = attn_out.transpose(-2, -1).contiguous().view(batch_size, -1, self.img_h,
                                                                self.img_w)

        attn_out = self.norm(attn_out)
        out_put = attn_out + x
        out_put = self.attn_out(out_put)
        return out_put
###################################################################################
class N2_Normalization(nn.Module):
    def __init__(self, upscale_factor):
        super(N2_Normalization, self).__init__()
        self.upscale_factor = upscale_factor
        self.avgpool = nn.AvgPool2d(upscale_factor)
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='nearest')
        self.epsilon = 1e-5

    def forward(self, x):
        out = self.avgpool(x) * self.upscale_factor ** 2  # sum pooling
        out = self.upsample(out)  #
        return torch.div(x, out + self.epsilon)


class GN2_Normalization(nn.Module):
    def __init__(self, upscale_factor=4):
        super(GN2_Normalization, self).__init__()
        self.N2 = N2_Normalization(upscale_factor=upscale_factor)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.N2(x)
        x2 = self.N2(x)
        x2 = self.act(x2)
        x3 = torch.mul(x1, x2)
        x4 = x1 + x3
        out = x4
        out = self.N2(out)
        return out


class Recover_from_density(nn.Module):
    def __init__(self, upscale_factor):
        super(Recover_from_density, self).__init__()
        self.upscale_factor = upscale_factor
        self.upsample = nn.Upsample(scale_factor=upscale_factor, mode='nearest')

    def forward(self, x, lr_img):
        out = self.upsample(lr_img)
        return torch.mul(x, out)