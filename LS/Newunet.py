import torch
import torch.nn as nn


# import tensorwatch as tw
# from torchviz import make_dot


def double_conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation
    )


# def double_transconv_block_3d(in_dim, out_dim, activation):
#     return nn.Sequential(
#         nn.ConvTranspose3d(in_dim, out_dim//2, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm3d(out_dim//2),
#         activation,
#         nn.Conv3d(out_dim//2, out_dim, kernel_size=3, stride=1, padding=1),
#         nn.BatchNorm3d(out_dim),
#         activation
#     )

def max_pooling_3d_noz():
    # this pooling don't change the number of z axis
    return nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))


def upscaling(x):
    return torch.nn.functional.interpolate(x, scale_factor=(1, 2, 2), mode='trilinear', align_corners=False)


class conv3d_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv3d_down, self).__init__()
        self.activation = nn.LeakyReLU()
        self.conv3d = double_conv_block_3d(in_ch, out_ch, self.activation)
        self.pool = max_pooling_3d_noz()

    def forward(self, x):
        conv_x = self.conv3d(x)
        out = self.pool(conv_x)
        return conv_x, out


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.activation = nn.LeakyReLU()
        self.conv3d = double_conv_block_3d(in_ch, out_ch, self.activation)
        self.upsample = upscaling
        self.outch = out_ch

    def forward(self, x):
        out = self.upsample(x)
        out1 = self.conv3d(out)
        return out1
class conv3d_up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv3d_up, self).__init__()
        self.activation = nn.LeakyReLU()
        self.conv3d = double_conv_block_3d(in_ch, out_ch, self.activation)
        self.upsample = upscaling
        self.outch = out_ch

    def forward(self, x):
        convx = self.conv3d(x)
        out = self.upsample(convx)
        return convx, out


class deep_supervison(nn.Module):
    def __init__(self, in_ch):
        super(deep_supervison, self).__init__()
        self.conv3d = nn.Conv3d(in_ch, 4, kernel_size=1, stride=1, padding=0)
        self.upsample = upscaling
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        conv3d = self.conv3d(x)
        conv3d = self.activation(conv3d)
        upscale = self.upsample(conv3d)
        return conv3d, upscale


class Insensee_3Dunet(nn.Module):
    def __init__(self, in_ch):
        super(Insensee_3Dunet, self).__init__()
        self.con3d_down1 = conv3d_down(in_ch, 26)
        self.con3d_down2 = conv3d_down(26, 52)
        self.con3d_down3 = conv3d_down(52, 104)
        self.con3d_down4 = conv3d_down(104, 208)
        self.con3d_brige = conv3d_down(208, 416)
        self.upconv5 = up_conv(416, 208)
        self.con3d_up4 = conv3d_up(416, 208)
        self.upconv4 = up_conv(208, 104)
        self.con3d_up3 = conv3d_up(208, 104)
        self.upconv3 = up_conv(104, 52)
        self.con3d_up2 = conv3d_up(104, 52)
        self.upconv2 = up_conv(52, 26)
        self.con3d_up1 = double_conv_block_3d(52, 26, nn.LeakyReLU())
        self.con3d_pre = nn.Conv3d(4, 4, kernel_size=1, stride=1, padding=0)
        self.deep_sup1 = deep_supervison(104)
        self.deep_sup2 = deep_supervison(52)
        self.deep_sup3 = deep_supervison(26)
        self.upsample = upscaling

    def forward(self, x):
        convx_1, out_1 = self.con3d_down1(x)
        convx_2, out_2 = self.con3d_down2(out_1)
        convx_3, out_3 = self.con3d_down3(out_2)
        convx_4, out_4 = self.con3d_down4(out_3)
        convx_bridge, out_5 = self.con3d_brige(out_4)
        # print(type(convx_bridge.shape[1]//2))

        up_conv5 = self.upconv5(convx_bridge)
        # print(convx_4.shape, up_conv5.shape)
        concat4 = torch.cat((convx_4, up_conv5), dim=1)
        # print(concat4.shape)

        convx_up4, out_6 = self.con3d_up4(concat4)
        up_conv4 = self.upconv4(convx_up4)
        # print(convx_3.shape, up_conv4.shape)
        concat3 = torch.cat((convx_3, up_conv4), dim=1)
        # print(concat3.shape)

        convx_up3, out_7 = self.con3d_up3(concat3)
        up_conv3 = self.upconv3(convx_up3)
        _, deep_up1 = self.deep_sup1(convx_up3)  # 1*1*1 conv3d => upscaling
        concat2 = torch.cat((convx_2, up_conv3), dim=1)
        # print(concat2.shape)

        convx_up2, out_8 = self.con3d_up2(concat2)
        up_conv2 = self.upconv2(convx_up2)
        deep_conv2, deep_up2 = self.deep_sup2(convx_up2)
        concat1 = torch.cat((convx_1, up_conv2), dim=1)
        # print(concat1.shape)

        convx_up1 = self.con3d_up1(concat1)
        deep_conv3, _ = self.deep_sup3(convx_up1)
        sum1 = torch.add(deep_up1, deep_conv2)
        # print(sum1.shape)
        sum1_up = self.upsample(sum1)
        sum2 = torch.add(sum1_up, deep_conv3)
        # print(sum2.shape)
        # pred = self.con3d_pre(sum2)

        return sum2


if __name__ == "__main__":
    x = torch.randn(1, 1, 5, 128, 128)
    Input = torch.randn(1, 4, 10, 128, 128)
    label = torch.randint(0, 4, (1, 10, 128, 128))
    model = Insensee_3Dunet(1)
    out = model(x)
    print(out.shape)
    # print(label.shape)
    # out = torch.argmax(out, dim=1).float()
    # print(out)
    Lossfun = torch.nn.CrossEntropyLoss()
    loss = Lossfun(Input, label)
    print(loss)
    # tw.draw_model(model, [1,1,5,128,128])
    # vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
    # vis_graph.view()
    # img.save(r'./Insensee_3D_unet.jpg')
    # print(out.shape)
    # print(x)
    # x = upscaling(x)
    # print(x)
