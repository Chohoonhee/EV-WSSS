import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.functional import interpolate as interpolate
from models.decoder import dec_deeplabv3_plus as Decoder
from models.resnet_official import resnet50

import pdb



# Deeplab 50
class Net(nn.Module):
    def __init__(self, input_c, output_c):  
        super().__init__()

        self.encoder = resnet50(in_ch = input_c, pretrained=False, fpn=True)
        self.decoder = Decoder(in_planes=2048, num_classes=output_c)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)   
            return self.decoder.forward(output)