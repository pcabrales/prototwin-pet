# From https://github.com/HuaxiangLiu/GCAU-Net/blob/main/GCAU-Net.py

from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
# from nnunet.network_architecture.neural_network import SegmentationNetwork
from models.nnFormer.neural_network import SegmentationNetwork

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Resconv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(Resconv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class EncoderNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_ch=3):
        super(EncoderNet, self).__init__()

        n = 1
        self.in_chanel = in_ch
        filters = [16*n, 32*n, 64*n, 128*n, 256*n, 512*n]
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inconv1 = Resconv_block(in_ch, filters[0])
        self.inconv2 = Resconv_block(filters[0], filters[1])
        self.inconv3 = Resconv_block(filters[1], filters[2])
        self.inconv4 = Resconv_block(filters[2], filters[3])
        self.inconv5 = Resconv_block(filters[3], filters[4])

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):

        out1 = self.inconv1(x)

        out2 = self.down(out1)
        out2 = self.inconv2(out2)

        out3 = self.down(out2)
        out3 = self.inconv3(out3)

        out4 = self.down(out3)
        out4 = self.inconv4(out4)

        out5 = self.down(out4)
        out5 = self.inconv5(out5)

        return out1,out2, out3, out4, out5

class MultiAttBlock(nn.Module):
    """
    Multi-scale Attention Block
    """
    def __init__(self, in_ch, out_ch):
        super(MultiAttBlock, self).__init__()
        self.pre_conv3 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.pre_conv5 = nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(in_ch * 3+out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    """
    f: current feature
    down: the features in the next layer
    """
    def forward(self, f,down):
        f_down = self.down(f)
        down_scale = F.interpolate(f_down, f.size()[2:], mode='bilinear',align_corners=True)
        #down_scale = torch.cat((f, down_scale), dim=1)
        f_down = down_scale + f

        f_scale3 = self.pre_conv3(f)
        f_down3 = self.pre_conv3(f_scale3)
        f_down3 = F.interpolate(f_down3, f.size()[2:], mode='bilinear', align_corners=True)
        f_down3 = f_scale3 + f_down3

        f_scale5 = self.pre_conv5(f)
        f_down5 = self.pre_conv3(f_scale5)
        f_down5 = F.interpolate(f_down5, f.size()[2:], mode='bilinear', align_corners=True)
        f_down5 = f_scale5 + f_down5

        up_scale = F.interpolate(down,f.size()[2:],mode = 'bilinear',align_corners=True)
        out = torch.cat((f_down, f_down3, f_down5, up_scale), dim=1)
        out = self.relu(self.bn(self.conv(out)))
        return out


class MultiAttBlockA(nn.Module):
    """
    Multi-scale Attention Block
    """
    def __init__(self, in_ch, out_ch):
        super(MultiAttBlockA, self).__init__()
        self.pre_conv3 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.pre_conv5 = nn.Conv2d(in_ch, in_ch, kernel_size=5, stride=1, padding=2)
        self.conv = nn.Conv2d(in_ch * 3+out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    """
    f: current feature
    down: the features in the next layer
    """
    def forward(self, f,down):
        f_down = self.down(f)
        down_scale = F.interpolate(f_down, f.size()[2:], mode='bilinear',align_corners=True)
        #down_scale = torch.cat((f, down_scale), dim=1)
        f_down = down_scale + f

        f_scale3 = self.pre_conv3(f)
        f_down3 = self.pre_conv3(f_scale3)
        f_down3 = F.interpolate(f_down3, f.size()[2:], mode='bilinear', align_corners=True)
        f_down3 = f_scale3 + f_down3

        f_scale5 = self.pre_conv5(f)
        f_down5 = self.pre_conv3(f_scale5)
        f_down5 = F.interpolate(f_down5, f.size()[2:], mode='bilinear', align_corners=True)
        f_down5 = f_scale5 + f_down5

        up_scale = F.interpolate(down,f.size()[2:],mode = 'bilinear',align_corners=True)
        out = torch.cat((f_down, f_down3, f_down5, up_scale), dim=1)
        out = self.relu(self.bn(self.conv(out)))
        return out

class FeaterExtra(nn.Module):


    def __init__(self, in_dim):
        super(FeaterExtra, self).__init__()
        self.chanel_in = in_dim
        self.activation = nn.ReLU(inplace=True)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out



class FeatureAggregration(nn.Module):
    def __init__(self, top_ch, left_ch,down_ch):
        super(FeatureAggregration, self).__init__()

        self.down_conv = nn.Sequential(
            nn.Conv2d(down_ch, top_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(top_ch),
            nn.ReLU(inplace=True))


        self.top_conv = nn.Sequential(
            nn.Conv2d(top_ch, top_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(top_ch))

        self.wx = nn.Sequential(
            nn.Conv2d(top_ch, top_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(top_ch))

        self.wy = nn.Sequential(
            nn.Conv2d(top_ch, top_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(top_ch))

        self.phi = nn.Sequential(
            nn.Conv2d(top_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1))

        self.left_conv = nn.Sequential(
            nn.Conv2d(left_ch, top_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(top_ch))

        self.conv = nn.Sequential(
            nn.Conv2d(top_ch+left_ch +top_ch, left_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(left_ch))

    def forward(self, top, left, down):

        down = self.down_conv(down)
        down = F.interpolate(down, top.size()[2:], mode='bilinear', align_corners=True)

        wx = self.wx(down)

        wy = self.wy(top)

        phi = F.relu(wx+wy)

        phi = self.phi(phi)
        phi = torch.sigmoid(phi)


        down = top*phi

        left = F.interpolate(left, top.size()[2:], mode='bilinear', align_corners=True)

        out = torch.cat((top, left, down), dim=1)
        out = self.conv(out)

        return out

class Featureconn(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, num_ch4,num_ch5):
        super(Featureconn, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_ch5, num_ch5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_ch5),
            nn.ReLU(inplace=True))

        self.catconv = nn.Sequential(
            nn.Conv2d(num_ch4+num_ch5, num_ch5, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_ch5),
            nn.ReLU(inplace=True))

        self.up =  nn.Upsample(scale_factor=2)

    def forward(self, top, center):
        center = self.conv(center)
        center = self.up(center)
        out = torch.cat((top, center), dim=1)
        out = self.catconv(out)
        return out


class GCAUNet(SegmentationNetwork):
    def __init__(self, in_ch=3, out_ch=1):
        super(GCAUNet, self).__init__()

        n=1
        filters = [16*n, 32*n, 64*n, 128*n, 256*n, 512*n]
        self.encoder = EncoderNet(in_ch)

        self.FE1 = FeaterExtra(filters[4])

        self.fa0 = FeatureAggregration(filters[3], filters[4], filters[4])
        self.fa1 = FeatureAggregration(filters[2], filters[3], filters[4])
        self.fa2 = FeatureAggregration(filters[1], filters[2], filters[4])
        self.fa3 = FeatureAggregration(filters[0], filters[1], filters[4])


        self.conv_layer5 = nn.Sequential(
            nn.Conv2d(filters[4], filters[4], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True))

        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(filters[4], filters[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True))

        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(filters[3], filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True))

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(filters[2], filters[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True))

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(filters[1], filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True))

        self.conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.do_ds = True
    def forward(self, x):
        layer1, layer2, layer3, layer4, layer5 = self.encoder(x)

        down_layer1 = self.FE1(layer5)


        out_layer5 = self.conv_layer5(down_layer1)

        out_decoder4 = self.fa0(layer4, out_layer5, down_layer1)
        out_decoder4 = self.conv_layer4(out_decoder4)

        out_decoder3 = self.fa1(layer3, out_decoder4, down_layer1)
        out_decoder3 = self.conv_layer3(out_decoder3)

        out_decoder2 = self.fa2(layer2, out_decoder3, down_layer1)
        out_decoder2 = self.conv_layer2(out_decoder2)

        out_decoder1 = self.fa3(layer1, out_decoder2, down_layer1)
        out_decoder1 = self.conv_layer1(out_decoder1)

        out = self.conv(out_decoder1)
        return out


def init(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)

def model_unet(model_input, in_channel=3, out_channel=1):
    model_test = model_input(in_channel, out_channel)
    return model_test
