import torchvision.models as models
import torch
import torch.nn as nn
import pdb
# torch.utils.model_zoo.load_url()

# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                #  std=[0.229, 0.224, 0.225])

        
# forward function for monkey patching
def new_forward(self, x):
    # See note [TorchScript super()]
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x1 = x
    x = self.layer2(x)
    x2 = x
    x = self.layer3(x)
    x3 = x
    x = self.layer4(x)
    x4 = x
    
    # x = self.avgpool(x)
    # x = torch.flatten(x, 1)
    # x = self.fc(x)
    
    # return x
    
    if self.fpn:
        return [x1, x2, x3, x4]
    else:
        return [x3, x4]

def resnet18(pretrained=False, fpn=False):
    if pretrained:
        model = models.resnet18(pretrained=True)
        print("Load pretrained weights.")
    else:
        model = models.resnet18()
        print("Not load pretrained weights.")
    type(model)._forward_impl = new_forward
    model.fpn = fpn
    return model


def resnet34(pretrained=False, fpn=False):
    if pretrained:
        model = models.resnet34(pretrained=True)
        print("Load pretrained weights.")
    else:
        model = models.resnet34()
        print("Not load pretrained weights.")
    type(model)._forward_impl = new_forward
    model.fpn = fpn
    return model


def resnet50(in_ch=3, pretrained=False, fpn=False):
    if pretrained:
        model = models.resnet50(pretrained=True)
        print("Load pretrained weights.")
    else:
        model = models.resnet50()
        print("Not load pretrained weights.")
    type(model)._forward_impl = new_forward
    model.fpn = fpn
    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7,\
                    stride=2, padding=3, bias=False)
    return model


def resnet101(pretrained=False, fpn=False):
    if pretrained:
        model = models.resnet101(pretrained=True)
        print("Load pretrained weights.")
    else:
        model = models.resnet101()
        print("Not load pretrained weights.")
    type(model)._forward_impl = new_forward
    model.fpn = fpn
    return model


def resnet152(pretrained=False, fpn=False):
    if pretrained:
        model = models.resnet152(pretrained=True)
        print("Load pretrained weights.")
    else:
        model = models.resnet152()
        print("Not load pretrained weights.")
    type(model)._forward_impl = new_forward
    model.fpn = fpn
    return model