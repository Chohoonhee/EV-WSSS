from audioop import bias
from configparser import BasicInterpolation
from cv2 import CAP_MSMF
from matplotlib.pyplot import get
from sklearn import feature_selection
import torch
from torch import nn
import numpy as np
import pdb
# from torch.nn.common_types import T

import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, first_dilation=None, dilation=1):
        super(ResBlock, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        if first_dilation == None: first_dilation = dilation

        self.bn_branch2a = nn.BatchNorm2d(in_channels)

        self.conv_branch2a = nn.Conv2d(in_channels, mid_channels, 3, stride,
                                       padding=first_dilation, dilation=first_dilation, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(mid_channels)

        self.conv_branch2b1 = nn.Conv2d(mid_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False, start_imd = False):

        branch2 = self.bn_branch2a(x)
        branch2 = F.relu(branch2)

        x_bn_relu = branch2

        if not self.same_shape:
            branch1 = self.conv_branch1(branch2)
        else:
            branch1 = x

        branch2 = self.conv_branch2a(branch2)
        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.conv_branch2b1(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu)

class ResBlock_bot(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout=0.):
        super(ResBlock_bot, self).__init__()

        self.same_shape = (in_channels == out_channels and stride == 1)

        self.bn_branch2a = nn.BatchNorm2d(in_channels)
        self.conv_branch2a = nn.Conv2d(in_channels, out_channels//4, 1, stride, bias=False)

        self.bn_branch2b1 = nn.BatchNorm2d(out_channels//4)
        self.dropout_2b1 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b1 = nn.Conv2d(out_channels//4, out_channels//2, 3, padding=dilation, dilation=dilation, bias=False)

        self.bn_branch2b2 = nn.BatchNorm2d(out_channels//2)
        self.dropout_2b2 = torch.nn.Dropout2d(dropout)
        self.conv_branch2b2 = nn.Conv2d(out_channels//2, out_channels, 1, bias=False)

        if not self.same_shape:
            self.conv_branch1 = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x, get_x_bn_relu=False,start_imd=False):
        if not start_imd:
            branch2 = self.bn_branch2a(x)
            branch2 = F.relu(branch2)
        else:
            branch2 = x
        x_bn_relu = branch2

        branch1 = self.conv_branch1(branch2)

        branch2 = self.conv_branch2a(branch2)

        branch2 = self.bn_branch2b1(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b1(branch2)
        branch2 = self.conv_branch2b1(branch2)

        branch2 = self.bn_branch2b2(branch2)
        branch2 = F.relu(branch2)
        branch2 = self.dropout_2b2(branch2)
        branch2 = self.conv_branch2b2(branch2)

        x = branch1 + branch2

        if get_x_bn_relu:
            return x, x_bn_relu

        return x

    def __call__(self, x, get_x_bn_relu=False, start_imd=False):
        return self.forward(x, get_x_bn_relu=get_x_bn_relu,start_imd=start_imd)

class Normalize():
    def __init__(self, mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img

def convert_mxnet_to_torch(filename):
    import mxnet

    save_dict = mxnet.nd.load(filename)

    renamed_dict = dict()

    bn_param_mx_pt = {'beta': 'bias', 'gamma': 'weight', 'mean': 'running_mean', 'var': 'running_var'}

    for k, v in save_dict.items():

        v = torch.from_numpy(v.asnumpy())
        toks = k.split('_')

        if 'conv1a' in toks[0]:
            renamed_dict['conv1a.weight'] = v

        elif 'linear1000' in toks[0]:
            pass

        elif 'branch' in toks[1]:

            pt_name = []

            if toks[0][-1] != 'a':
                pt_name.append('b' + toks[0][-3] + '_' + toks[0][-1])
            else:
                pt_name.append('b' + toks[0][-2])

            if 'res' in toks[0]:
                layer_type = 'conv'
                last_name = 'weight'

            else:  # 'bn' in toks[0]:
                layer_type = 'bn'
                last_name = bn_param_mx_pt[toks[-1]]

            pt_name.append(layer_type + '_' + toks[1])

            pt_name.append(last_name)

            torch_name = '.'.join(pt_name)
            renamed_dict[torch_name] = v

        else:
            last_name = bn_param_mx_pt[toks[-1]]
            renamed_dict['bn7.' + last_name] = v

    return renamed_dict

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class Net_moco(nn.Module):
    def __init__(self, IC=100, C=20):
        super(Net_moco, self).__init__()

        self.conv1 = nn.Conv2d(IC, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.1)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.1)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, 384, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(384)
        self.fc9 = nn.Conv2d(384, C, 1, bias=False)
        # self.fc10 = nn.Conv2d(C, D, 1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)
        # torch.nn.init.xavier_uniform_(self.fc10.weight)

        # self.temp = torch.nn.Parameter(torch.ones(512,512),requires_grad=True).cuda()

        self.from_scratch_layers = []
        self.not_training = []

    def forward(self, x):
        
        x = self.conv1(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x = self.b4(x)  # B x 512 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x,feat0 = self.b5(x,get_x_bn_relu=True)  # B x 1024 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x,feat1 = self.b6(x,get_x_bn_relu=True)
        x,feat2 = self.b7(x,get_x_bn_relu=True)

        x = F.relu(self.bn7(x))

        x =self.fc8(x)  # B x D x 56 x 56 #기존
        feat3 = x #256
        # feat3 = x #256
        cam = self.fc9(F.relu(x))
        pred = self.avgpool(cam).squeeze(3).squeeze(2)


        return cam, pred 

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class Net_moco2(nn.Module):
    def __init__(self, D=256, C=20):
        super(Net_moco2, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)
        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, D, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(D)
        self.fc9 = nn.Conv2d(D, C, 1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.3)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)

        self.from_scratch_layers = []
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]

    def forward(self, x, cam=None):
        if cam==None:
            x = self.conv1a(x)

            x = self.b2(x)
            x = self.b2_1(x)
            x = self.b2_2(x)

            x = self.b3(x)
            x = self.b3_1(x)
            x = self.b3_2(x)

            x = self.b4(x)  # B x 512 x 56 x 56
            x = self.b4_1(x)
            x = self.b4_2(x)
            x = self.b4_3(x)

            x = self.b4_4(x)
            x = self.b4_5(x)

            x = self.b5(x)  # B x 1024 x 56 x 56
            x = self.b5_1(x)
            x = self.b5_2(x)

            x = self.b6(x)
            x = self.b7(x)

            x = F.relu(self.bn7(x))

            feat = F.relu(self.fc8(x))  # B x D x 56 x 56 #기존
            # feat = self.fc8(x)  # B x D x 56 x 56
            cam = self.fc9(feat)
            pred = self.avgpool(cam).squeeze(3).squeeze(2)

            return feat, cam, pred

        else:
            
            mask = torch.randint_like(cam, 3)

            mask0 = 1-cam
            mask1 = 1-cam
            mask2 = 1-cam

            x = self.conv1a(x)

            x = self.b2(x)
            x = self.b2_1(x)
            x = self.b2_2(x)

            x = self.b3(x)
            x = self.b3_1(x)
            x = self.b3_2(x)

            x = self.b4(x)  # B x 512 x 56 x 56
            x = mask0.detach()*x
            x = self.b4_1(x)
            x = self.b4_2(x)
            x = self.b4_3(x)

            x = self.b4_4(x)
            x = self.b4_5(x)

            x = self.b5(x)  # B x 1024 x 56 x 56
            x = mask1.detach()*x
            x = self.b5_1(x)
            x = self.b5_2(x)

            x = self.b6(x)
            x = mask2.detach()*x
            x = self.b7(x)

            x = F.relu(self.bn7(x))

            feat = F.relu(self.fc8(x))  # B x D x 56 x 56 #기존

            cam = self.fc9(feat)
            pred = self.avgpool(cam).squeeze(3).squeeze(2)

            return cam, pred


    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class Net_recon(nn.Module):
    def __init__(self):
        super(Net_recon, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.2)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.2)
        self.bn7 = nn.BatchNorm2d(4096)

        self.fc8 = nn.Conv2d(4096, 256, 1, bias=False)
        self.fc9 = nn.Conv2d(256, 20, 1, bias=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(0.5)
        #

        # best
        self.recon1 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(212,128,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(128),
            )

        self.recon2 = nn.Sequential(
            # nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(128,64,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.BatchNorm2d(64),
            )
        self.recon3 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1,bias=False),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.2,inplace=True),
            # nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(32,3,3,1,1,bias=False),
            # nn.Hardtanh(-1.0, 1.0) #best
            # nn.Tanh()
        )

      
        self.dr1 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.dr2 = torch.nn.Conv2d(1024, 128, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc9.weight)
        torch.nn.init.kaiming_normal_(self.dr1.weight)
        torch.nn.init.kaiming_normal_(self.dr2.weight)

        self.from_scratch_layers = []#self.recon1,self.recon2,self.recon3]
        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
    
    def recon_decoder(self,feat):
        r1 = self.recon1(feat)
        r2 = self.recon2(r1)
        recon_img = self.recon3(r2) 
        return recon_img

    def forward(self, img):

        _, _, H, W = img.shape

        x = self.conv1a(img)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        x, conv3 = self.b4(x, get_x_bn_relu=True)  # B x 512 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)

        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x,get_x_bn_relu=True)  # B x 1024 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x,get_x_bn_relu=True)  # B x 2048 x 56 x 56

        x = self.b7(x)
        x = F.relu(self.bn7(x))
        feat = F.relu(self.fc8(x))

        cam = self.fc9(feat)

        conv4 = self.dr1(conv4)
        conv5 = self.dr2(conv5)

        feat_cat = torch.cat([conv4,conv5,cam],dim=1) # 128+64+20 => 212
        # feat_cat = torch.cat([conv5,feat],dim=1) # 128+64+20 => 212
        recon_img = self.recon_decoder(feat_cat)
       
        out = self.avgpool(cam).squeeze(3).squeeze(2)

        # return cam, out, [conv3,feat]
        # return cam, out, [recon_img,feat_cat, x]
        return cam, out, feat_cat

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm) or isinstance(m,nn.ConvTranspose2d)or isinstance(m,nn.ConvTranspose2d)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class ABF(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super(ABF, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channel),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channel, out_channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channel),
        )
        if fuse:
            self.att_conv = nn.Sequential(
                    nn.Conv2d(mid_channel*2, 2, kernel_size=1),
                    nn.Sigmoid(),
                )
        else:
            self.att_conv = None
        nn.init.kaiming_uniform_(self.conv1[0].weight, a=1)  # pyre-ignore
        nn.init.kaiming_uniform_(self.conv2[0].weight, a=1)  # pyre-ignore

    def forward(self, x, y, shape=None, out_shape=None):
        n,_,h,w = x.shape
        # transform student features
        x = self.conv1(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (h,w), mode="bilinear")
            # fusion
            z = torch.cat([x, y], dim=1)
            z = self.att_conv(z)
            x = (x * z[:,0].view(n,1,h,w) + y * z[:,1].view(n,1,h,w))
        # output 
        # if x.shape[-1] != y.shape[-1]:
            # x = F.interpolate(x, (out_shape, out_shape), mode="Bilinear")
        y = self.conv2(x)
        return y, x

class ABF2(nn.Module):
    def __init__(self,in_ch1,in_ch2):
        super(ABF2,self).__init__()
        
        self.att_p = nn.Sequential(
                    nn.Conv2d(in_ch1*2, 2, 3,1,1,bias=False),
                    nn.Sigmoid(),
                )
        self.att_n = nn.Sequential(
                    nn.Conv2d(in_ch1*2, 2, 3,1,1,bias=False),
                    nn.Sigmoid(),
                )
    
    def forward(self,x,y):
        n,c,h,w = y.size()
        # x = F.interpolate(x, (h,w), mode="bilinear",align_corners=False)
        # y = self.dc2(y)
        xy_p = torch.cat([F.relu(x),F.relu(y)],dim=1)
        xy_n = torch.cat([F.relu(-x),F.relu(-y)],dim=1)
        att_p = self.att_p(xy_p)
        att_n = self.att_n(xy_n)
        final = (F.relu(x)*att_p[:,0].view(n,1,h,w)+F.relu(y)*att_p[:,1].view(n,1,h,w))/2
        final -=(F.relu(-x)*att_n[:,0].view(n,1,h,w)+F.relu(-y)*att_n[:,1].view(n,1,h,w))/2

        return final


class Net_dl(nn.Module):
    def __init__(self):
        super(Net_dl, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)
        self.bn7 = nn.BatchNorm2d(4096)

        self.dropout = nn.Dropout2d(0.5)

        self.fc8_b8 = nn.Conv2d(4096, 512, 3, dilation=12, padding=12, bias=True)

        self.fc8_b9 = nn.Conv2d(512, 21, 3, dilation=12, padding=12, bias=True)

        self.not_training = []#self.conv1a,self.b2,self.b2_1,self.b2_2] #original []
        self.from_scratch_layers = [self.fc8_b8, self.fc8_b9] #original version on

        torch.nn.init.xavier_uniform_(self.fc8_b8.weight)
        torch.nn.init.xavier_uniform_(self.fc8_b9.weight)

        self.normalize = Normalize()

    def forward(self, x):

        _, _, H, W = x.shape

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        # x = self.b4(x)
        x, conv3 = self.b4(x, get_x_bn_relu=True)  # B x 256 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)  # B x 512 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)  # B x 1024 x 56 x 56

        x = self.b7(x)
        x = F.relu(self.bn7(x))#originally no inplace
        x = F.relu(self.fc8_b8(x))#originally no inplace

        x = self.fc8_b9(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_10x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' in name:
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' not in name:
                yield param

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


# class _ASPP(nn.Module):
#     """
#     Atrous spatial pyramid pooling (ASPP)
#     """

#     def __init__(self, in_ch, out_ch, rates):
#         super(_ASPP, self).__init__()
#         for i, rate in enumerate(rates):
#             self.add_module(
#                 "c{}".format(i),
#                 nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
#             )

#         for m in self.children():
#             nn.init.normal_(m.weight, mean=0, std=0.01)
#             nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         return sum([stage(x) for stage in self.children()])


class Net_dl_V2(nn.Module):
    def __init__(self):
        super(Net_dl_V2, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)
        self.bn7 = nn.BatchNorm2d(4096)

        self.dropout = nn.Dropout2d(0.5)

        self.fc8_1 = nn.Conv2d(4096, 512, 3, dilation=12, padding=12, bias=True)

        self.fc8_d6 = nn.Conv2d(512, 21, 3, 1, padding=6, dilation=6, bias=True)
        self.fc8_d12 = nn.Conv2d(512, 21, 3, 1, padding=12, dilation=12, bias=True)
        self.fc8_d18 = nn.Conv2d(512, 21, 3, 1, padding=18, dilation=18, bias=True)
        self.fc8_d24 = nn.Conv2d(512, 21, 3, 1, padding=24, dilation=24, bias=True)

        self.not_training = []#self.conv1a] #original []
        self.from_scratch_layers = [self.fc8_1,self.fc8_d6, self.fc8_d12, self.fc8_d18, self.fc8_d24] #original version on

        torch.nn.init.xavier_uniform_(self.fc8_d6.weight)
        torch.nn.init.xavier_uniform_(self.fc8_d12.weight)
        torch.nn.init.xavier_uniform_(self.fc8_d18.weight)
        torch.nn.init.xavier_uniform_(self.fc8_d24.weight)

        self.normalize = Normalize()

    def forward(self, x):

        _, _, H, W = x.shape

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        # x = self.b4(x)
        x, conv3 = self.b4(x, get_x_bn_relu=True)  # B x 256 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)  # B x 512 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)  # B x 1024 x 56 x 56

        x = self.b7(x)
        x = F.relu(self.bn7(x))#originally no inplace
        x = F.relu(self.fc8_1(x))#originally no inplace

        x = self.fc8_d6(x) + self.fc8_d12(x) + self.fc8_d18(x) + self.fc8_d24(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_10x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' in name:
                yield param

    def get_1x_lr_params(self):
        for name, param in self.named_parameters():
            if 'fc8' not in name:
                yield param

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups



class Net_dl_cls(nn.Module):
    def __init__(self):
        super(Net_dl_cls, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)
        self.bn7 = nn.BatchNorm2d(4096)

        self.cls_head = nn.Conv2d(4096,20,1,1,0,bias=False)

        self.dropout = nn.Dropout2d(0.5)

        self.fc8_b8 = nn.Conv2d(4096, 512, 3, dilation=12, padding=12, bias=True)

        self.fc8_b9 = nn.Conv2d(512, 21, 3, dilation=12, padding=12, bias=True)

        self.not_training = []#self.conv1a] #original []
        self.from_scratch_layers = [self.fc8_b8, self.fc8_b9] #original version on

        torch.nn.init.xavier_uniform_(self.fc8_b8.weight)
        torch.nn.init.xavier_uniform_(self.fc8_b9.weight)

        self.normalize = Normalize()

    def forward(self, x):

        B, _, H, W = x.shape

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        # x = self.b4(x)
        x, conv3 = self.b4(x, get_x_bn_relu=True)  # B x 256 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)  # B x 512 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)  # B x 1024 x 56 x 56

        x = self.b7(x)
        x = F.relu(self.bn7(x))#originally no inplace
        cls = self.cls_head(x)
        cls = F.adaptive_avg_pool2d(cls,(1,1)).view(B,20)
        x = F.relu(self.fc8_b8(x))#originally no inplace

        x = self.fc8_b9(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x,cls

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1a = nn.Conv2d(3, 64, 3, padding=1, bias=False)

        self.b2 = ResBlock(64, 128, 128, stride=2)
        self.b2_1 = ResBlock(128, 128, 128)
        self.b2_2 = ResBlock(128, 128, 128)

        self.b3 = ResBlock(128, 256, 256, stride=2)
        self.b3_1 = ResBlock(256, 256, 256)
        self.b3_2 = ResBlock(256, 256, 256)

        self.b4 = ResBlock(256, 512, 512, stride=2)
        self.b4_1 = ResBlock(512, 512, 512)
        self.b4_2 = ResBlock(512, 512, 512)
        self.b4_3 = ResBlock(512, 512, 512)
        self.b4_4 = ResBlock(512, 512, 512)
        self.b4_5 = ResBlock(512, 512, 512)

        self.b5 = ResBlock(512, 512, 1024, stride=1, first_dilation=1, dilation=2)
        self.b5_1 = ResBlock(1024, 512, 1024, dilation=2)
        self.b5_2 = ResBlock(1024, 512, 1024, dilation=2)

        self.b6 = ResBlock_bot(1024, 2048, stride=1, dilation=4, dropout=0.3)

        self.b7 = ResBlock_bot(2048, 4096, dilation=4, dropout=0.5)

        self.bn7 = nn.BatchNorm2d(4096)

        self.not_training = [self.conv1a]

        self.normalize = Normalize()

        return

    def forward(self, x):
        return self.forward_as_dict(x)['conv6']

    def forward_as_dict(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        #x = self.b4(x)
        x, conv3 = self.b4(x, get_x_bn_relu=True)  # B x 256 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x, conv4 = self.b5(x, get_x_bn_relu=True)  # B x 512 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)  # B x 1024 x 56 x 56

        x = self.b7(x)
        conv6 = F.relu(self.bn7(x)) # B x 4096 x 56 x 56

        return dict({'conv3': conv3, 'conv4': conv4, 'conv5': conv5, 'conv6': conv6})

    def forward_conv5(self, x):

        x = self.conv1a(x)

        x = self.b2(x)
        x = self.b2_1(x)
        x = self.b2_2(x)

        x = self.b3(x)
        x = self.b3_1(x)
        x = self.b3_2(x)

        #x = self.b4(x)
        x = self.b4(x)  # B x 256 x 56 x 56
        x = self.b4_1(x)
        x = self.b4_2(x)
        x = self.b4_3(x)
        x = self.b4_4(x)
        x = self.b4_5(x)

        x = self.b5(x)  # B x 512 x 56 x 56
        x = self.b5_1(x)
        x = self.b5_2(x)

        x, conv5 = self.b6(x, get_x_bn_relu=True)  # B x 1024 x 56 x 56

        return x, conv5

    def train(self, mode=True):

        super().train(mode)

        for layer in self.not_training:

            if isinstance(layer, torch.nn.Conv2d):
                layer.weight.requires_grad = False

            elif isinstance(layer, torch.nn.Module):
                for c in layer.children():
                    c.weight.requires_grad = False
                    if c.bias is not None:
                        c.bias.requires_grad = False

        for layer in self.modules():

            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.eval()
                layer.bias.requires_grad = False
                layer.weight.requires_grad = False

        return