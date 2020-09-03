# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image
from torch.nn.functional import sigmoid


class Double_conv(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=1),
            # nn.InstanceNorm2d(out_ch),
            nn.GroupNorm(num_groups=4, num_channels=out_ch, eps=0, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1),
            nn.GroupNorm(num_groups=4, num_channels=out_ch, eps=0, affine=False),
            # nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Conv_down(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_down, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        pool_x = self.pool(x)
        return pool_x, x


class Conv_up(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_up, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)
        self.up = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0, stride=1)
        self.interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1, x2):
        x1 = self.interp(x1)
        x1 = self.up(x1)
        x1_dim = x1.size()[2]
        x2 = extract_img(x1_dim, x2)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv(x1)
        return x1


class Conv_up_nl(nn.Module):
    '''(conv => ReLU) * 2 => MaxPool2d'''

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int) : input channel
            out_ch(int) : output channel
        """
        super(Conv_up_nl, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)
        self.interp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x1):
        x1 = self.interp(x1)
        x1 = self.conv(x1)
        return x1


def extract_img(size, in_tensor):
    """
    Args:
        size(int) : size of cut
        in_tensor(tensor) : tensor to be cut
    """
    dim1, dim2 = in_tensor.size()[2:]
    in_tensor = in_tensor[:, :, int((dim1 - size) / 2):int((dim1 + size) / 2),
                int((dim2 - size) / 2):int((dim2 + size) / 2)]
    return in_tensor


class deep_supervison(nn.Module):
    def __init__(self, in_ch):
        super(deep_supervison, self).__init__()
        self.conv2d = nn.Conv2d(in_ch, 4, kernel_size=1, stride=1, padding=0)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        conv2d = self.conv2d(x)
        conv2d = self.activation(conv2d)
        upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(conv2d)
        return conv2d, upscale


class DeepSupervision_U_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepSupervision_U_Net, self).__init__()
        self.Conv_down1 = Conv_down(in_channels, 64)
        self.Conv_down2 = Conv_down(64, 128)
        self.Conv_down3 = Conv_down(128, 256)
        self.Conv_down4 = Conv_down(256, 512)
        self.Conv_down5 = Conv_down(512, 1024)

        self.Conv_up1 = Conv_up(1024, 512)
        self.Conv_up2 = Conv_up(512, 256)
        self.Conv_up3 = Conv_up(256, 128)
        self.Conv_out1 = nn.Conv2d(128, out_channels, 1, padding=0, stride=1)
        self.Conv_up4 = Conv_up(128, 64)
        self.Conv_out2 = nn.Conv2d(64, out_channels, 1, padding=0, stride=1)
        self.deep3 = deep_supervison(256)
        self.deep2 = deep_supervison(128)
        self.deep1 = deep_supervison(64)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)


    def forward(self, x):
        x, conv1 = self.Conv_down1(x)
        # print("dConv1 => down1|", x.shape)
        x, conv2 = self.Conv_down2(x)
        # print("dConv2 => down2|", x.shape)
        x, conv3 = self.Conv_down3(x)
        # print("dConv3 => down3|", x.shape)
        x, conv4 = self.Conv_down4(x)
        # print("dConv4 => down4|", x.shape)
        _, x = self.Conv_down5(x)
        # print("dConv5|", x.shape)
        x = self.Conv_up1(x, conv4)
        # print("up1 => uConv1|", x.shape)
        x = self.Conv_up2(x, conv3)
        # print("up2 => uConv2|", x.shape)
        # print(x.shape[1])
        deep_conv3, deep_up3 = self.deep3(x)
        x = self.Conv_up3(x, conv2)
        # print(x.shape[1])
        deep_conv2, deep_up2 = self.deep2(x)
        # # print("up3 => uConv3|", x.shape)
        # x1 = self.Conv_out1(x)
        x = self.Conv_up4(x, conv1)
        deep_conv1, _ = self.deep1(x)
        # print(deep_up3.shape,deep_conv2.shape)
        sum1 = torch.add(deep_up3, deep_conv2)
        # print(sum1.shape)
        sum1 = self.upsample(sum1)
        x2 = self.Conv_out2(x)
        # print(x2.shape)
        sum2 = torch.add(sum1, deep_conv1)
        return sum2


class CleanU_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CleanU_Net, self).__init__()
        self.Conv_down1 = Conv_down(in_channels, 64)
        self.Conv_down2 = Conv_down(64, 128)
        self.Conv_down3 = Conv_down(128, 256)
        self.Conv_down4 = Conv_down(256, 512)
        self.Conv_down5 = Conv_down(512, 1024)

        self.Conv_up1 = Conv_up(1024, 512)
        self.Conv_up2 = Conv_up(512, 256)
        self.Conv_up3 = Conv_up(256, 128)
        self.Conv_out1 = nn.Conv2d(128, out_channels, 1, padding=0, stride=1)
        self.Conv_up4 = Conv_up(128, 64)
        self.Conv_out2 = nn.Conv2d(64, out_channels, 1, padding=0, stride=1)

    def forward(self, x):
        x, conv1 = self.Conv_down1(x)
        # print("dConv1 => down1|", x.shape)
        x, conv2 = self.Conv_down2(x)
        # print("dConv2 => down2|", x.shape)
        x, conv3 = self.Conv_down3(x)
        # print("dConv3 => down3|", x.shape)
        x, conv4 = self.Conv_down4(x)
        # print("dConv4 => down4|", x.shape)
        _, x = self.Conv_down5(x)
        # print("dConv5|", x.shape)
        x = self.Conv_up1(x, conv4)
        # print("up1 => uConv1|", x.shape)
        x = self.Conv_up2(x, conv3)
        # print("up2 => uConv2|", x.shape)
        x = self.Conv_up3(x, conv2)
        # print("up3 => uConv3|", x.shape)
        x1 = self.Conv_out1(x)
        x = self.Conv_up4(x, conv1)
        x2 = self.Conv_out2(x)
        return x2


if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 1, 512, 512)
    model = DeepSupervision_U_Net(1, 4)
    x = model(im)
    print(x.shape)
    del model
    del x
    # print(x.shape)
