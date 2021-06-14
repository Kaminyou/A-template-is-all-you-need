import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision import models

class ResNetEncoder(nn.Module):

    def __init__(self, name='resnet18', low_dim=128, batch_norm=False, pretrained=False):
        super(ResNetEncoder, self).__init__()
        
        builder = getattr(models, name)
        resnet = builder(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # projection MLP
        dim = resnet.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim) if batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(dim, low_dim),
        )

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                    
    def forward(self, x):
        """
        Input:
            x: a batch of images
        
        Output:
            proj: the projections of x for contrastive learning
            feat: the features of x for downstream task
                  in this case, the input (code) of the decoder
        """
        x = self.encoder(x)
        feat = x.view(x.size(0), -1)   

        proj = self.fc(feat)
        proj = F.normalize(proj) 
        
        return proj, feat