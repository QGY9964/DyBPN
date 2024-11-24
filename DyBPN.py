
import torch
import torch.nn as nn
from basic import DAF, MCA, SelfAttention, GN2_Normalization, Recover_from_density


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, ):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.bn = torch.nn.InstanceNorm2d(output_size)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.bn(self.conv(x))

        return self.act(out)


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True,
                 ):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.bn = torch.nn.InstanceNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.bn(self.deconv(x))

        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class GBPMs(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, scale_factor):
        super(GBPMs, self).__init__()

        self.attn = MCA(d_model=base_filter, head=4)

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # CSAB
        self.feat0 = ConvBlock(num_channels, feat, 3, 1, 1)
        self.feat1 = ConvBlock(feat, base_filter, 1, 1, 0)
        self.self_attn1 = SelfAttention(base_filter, base_filter, head=1)

        # DBPM
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = D_DownBlock(base_filter, kernel, stride, padding, 2)
        self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 3)
        self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)
        self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 4)
        self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
        self.down5 = D_DownBlock(base_filter, kernel, stride, padding, 5)
        self.up6 = D_UpBlock(base_filter, kernel, stride, padding, 5)
        self.down6 = D_DownBlock(base_filter, kernel, stride, padding, 6)
        self.up7 = D_UpBlock(base_filter, kernel, stride, padding, 6)
        self.down7 = D_DownBlock(base_filter, kernel, stride, padding, 7)
        self.up8 = D_UpBlock(base_filter, kernel, stride, padding, 7)
        self.down8 = D_DownBlock(base_filter, kernel, stride, padding, 8)
        self.up9 = D_UpBlock(base_filter, kernel, stride, padding, 8)
        self.down9 = D_DownBlock(base_filter, kernel, stride, padding, 9)
        self.up10 = D_UpBlock(base_filter, kernel, stride, padding, 9)

    def forward(self, x):

        # CSAB
        x = self.feat0(x)
        x = self.feat1(x)
        inp = x
        x = self.self_attn1(x) + inp

        # GBPM1
        h1_1 = self.up1(x)
        l1_1 = self.down1(h1_1)
        h1_2 = self.up2(l1_1)
        concat_h1 = torch.cat((h1_2, h1_1), 1)

        l2_1 = self.down2(concat_h1)
        concat_l2 = torch.cat((l2_1, l1_1), 1)
        h2_1 = self.up3(concat_l2)
        concat_h2 = torch.cat((h2_1, concat_h1), 1)
        l2_2 = self.down3(concat_h2)

        l2_2 = self.attn(x, l2_2)

        # GBPM2
        concat_l3 = torch.cat((l2_2, concat_l2), 1)
        h3_1 = self.up4(concat_l3)
        concat_h3 = torch.cat((h3_1, concat_h2), 1)
        l3_1 = self.down4(concat_h3)
        concat_l4 = torch.cat((l3_1, concat_l3), 1)
        h3_2 = self.up5(concat_l4)

        concat_h4 = torch.concat((h3_2, concat_h3), 1)
        l4_1 = self.down5(concat_h4)
        concat_l5 = torch.cat((l4_1, concat_l4), 1)
        h4_1 = self.up6(concat_l5)
        concat_h5 = torch.cat((h4_1, concat_h4), 1)
        l4_2 = self.down6(concat_h5)

        l4_2 = self.attn(l2_2, l4_2)

        concat_l6 = torch.cat((l4_2, concat_l5), 1)
        h5_1 = self.up7(concat_l6)
        concat_h6 = torch.cat((h5_1, concat_h5), 1)
        l5_1 = self.down7(concat_h6)
        concat_l7 = torch.cat((l5_1, concat_l6), 1)
        h5_2 = self.up8(concat_l7)

        # GBPM3
        concat_h7 = torch.cat((h5_2, concat_h6), 1)
        l6_1 = self.down8(concat_h7)
        concat_l8 = torch.cat((l6_1, concat_l7), 1)
        h6_1 = self.up9(concat_l8)
        concat_h8 = torch.cat((h6_1, concat_h7), 1)
        l6_2 = self.down9(concat_h8)

        l6_2 = self.attn(l4_2, l6_2)

        concat_l9 = torch.cat((l6_2, concat_l8), 1)  #
        h7_1 = self.up10(concat_l9)
        h7_2 = torch.cat((h7_1, concat_h8), 1)

        x = h7_2
        return x


class DyBPN(nn.Module):
    def __init__(self, base_channels, in_channels, scaler_X, scaler_Y, img_width, img_height, out_channels, ext_dim=7,
                 ):

        super().__init__()
        self.base_channel = base_channels
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.ext_dim = ext_dim
        self.img_width = img_width
        self.img_height = img_height

        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

        self.embed_day = nn.Embedding(8, 2)  # Monday: 1, Sunday:7, ignore 0, thus use 8
        self.embed_hour = nn.Embedding(24, 3)  # hour range [0, 23]
        self.embed_weather = nn.Embedding(18, 3)  # ignore 0, thus use 18

        self.ext2lr = nn.Sequential(
            nn.Linear(12, 128),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            nn.Linear(128, img_width * img_height),
            nn.ReLU(inplace=True)
        )

        self.ext2hr = nn.Sequential(
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.InstanceNorm2d(4),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 4, 3, 1, 1),
            nn.InstanceNorm2d(4),
            nn.PixelShuffle(upscale_factor=2),
            nn.ReLU(inplace=True)
        )

        conv1_in = in_channels + 1
        self.daf = DAF(conv1_in)

        self.conv_1 = nn.Sequential(nn.Conv2d(conv1_in, base_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(inplace=True))

        self.N2_norm = GN2_Normalization(4)
        self.recover = Recover_from_density(4)

        self.bp = GBPMs(num_channels=base_channels, base_filter=4 * base_channels, feat=2 * base_channels, num_stages=7,
                      scale_factor=4)

        self.out_layer = nn.Conv2d(1281, out_channels, 1, 1)  #

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, ext):
        inp = x

        ext_out1 = self.embed_day(ext[:, 4].long().view(-1, 1)).view(-1, 2)
        ext_out2 = self.embed_hour(ext[:, 5].long().view(-1, 1)).view(-1, 3)
        ext_out3 = self.embed_weather(ext[:, 6].long().view(-1, 1)).view(-1, 3)
        ext_out4 = ext[:, :4]
        ext_lr1 = self.ext2lr(torch.cat([ext_out1, ext_out2, ext_out3, ext_out4], dim=1)).view(-1, 1, self.img_width,
                                                                                               self.img_height)

        out0_2 = self.daf(inp, ext)
        bp = self.bp(out0_2)
        ext_hr = self.ext2hr(ext_lr1)
        out = torch.concat([bp, ext_hr], dim=1)
        out = self.out_layer(out)
        out = self.N2_norm(out)
        out = self.recover(out, x * self.scaler_X / self.scaler_Y)

        return out
