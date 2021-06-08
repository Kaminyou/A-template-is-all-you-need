import torch
import torchvision

class Encoder(torch.nn.Module):
    def __init__(self, latent_size):
        super(Encoder, self).__init__()
        self.encoder = torchvision.models.resnet18(pretrained=True)
        self.layers = list(self.encoder.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*self.layers)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.encoder.fc = torch.nn.Linear(self.encoder.fc.in_features,latent_size)

    def forward(self, img, training=True):
        latent_vec = self.feature_extractor(img)
        latent_vec = self.encoder.fc(torch.squeeze(latent_vec))
        return latent_vec