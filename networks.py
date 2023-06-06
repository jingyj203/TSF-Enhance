import torchvision.models as models
import torch.nn as nn
import torch

class Feature(nn.Module):
    def __init__(self, model='resnet50',dim3=1024,dim4=1024):
        nn.Module.__init__(self)
        self.model = model
        self.base = models.__dict__[model](pretrained=True)
        self.GMP = nn.AdaptiveMaxPool2d((1, 1))
        self.linear2 = torch.nn.Linear(512,341)
        self.linear3 = torch.nn.Linear(1024,dim3)
        self.linear4 = torch.nn.Linear(2048,dim4)
        self.lnorm2 = nn.LayerNorm(512,elementwise_affine=False).cuda()
        self.lnorm3 = nn.LayerNorm(1024, elementwise_affine=False).cuda()
        self.lnorm4 = nn.LayerNorm(2048, elementwise_affine=False).cuda()
    def forward(self, x):
        # x.shape   torch.Size([32, 3, 256, 256])
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        # x.shape   torch.Size([32, 64, 64, 64])
        x1 = self.base.layer1(x)
        # x1.shape  torch.Size([32, 256, 64, 64])
        x2 = self.base.layer2(x1)
        x3 = self.base.layer3(x2)
        x4 = self.base.layer4(x3)

        embedding2 = self.FEM(x2)
        embedding3 = self.FEM(x3)
        embedding4 = self.FEM(x4)

        return  embedding2, embedding3, embedding4
        #return embedding3,embedding4
    def FEM(self,features):
        features_1 = self.GMP(features)
        features = features_1
        features = features.reshape(features.size(0), -1)
        if features.shape[1] == 1024:
            features = self.lnorm3(features)
            embedding = self.linear3(features)
        elif features.shape[1] == 2048:
            features = self.lnorm4(features)
            embedding = self.linear4(features)
        else:
            features = self.lnorm2(features)
            embedding = self.linear2(features)
        return embedding


class Feat_resnet50_max_n(Feature):
     def __init__(self,dim3=1024, dim4=1024):
        Feature.__init__(self, model='resnet50',dim3=dim3,dim4=dim4)




