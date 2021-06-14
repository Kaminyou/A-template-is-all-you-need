import torch
import torchvision
import math
import torch.nn.functional as F
from torchvision import models

class Encoder(nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.layers = list(self.encoder.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*self.layers)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.encoder.fc = torch.nn.Linear(self.encoder.fc.in_features,latent_size)

    def forward(self, img, training=True):
        latent_vec = self.feature_extractor(img)
        latent_vec = self.encoder.fc(torch.squeeze(latent_vec))
        return latent_vec

class _Encoder(nn.Module):
    def __init__(self, name='resnet18', latent_size=256, pretrained=True):
        super(Encoder, self).__init__()
        
        builder = getattr(models, name)
        resnet = builder(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, latent_size)

        if not pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                    
    def forward(self, x):
        x = self.encoder(x)
        feat = x.view(x.size(0), -1)
        latent_vec = self.fc(feat)
        return latent_vec