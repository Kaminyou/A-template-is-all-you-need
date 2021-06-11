import torch
from torch import nn
import torch.nn.functional as F


class SimCLR(nn.Module):

    def __init__(self, base_encoder, low_dim=128, T=0.07, pretrained=True, **kwargs):
        super(SimCLR, self).__init__()
        
        self.encoder = base_encoder(low_dim=low_dim, pretrained=pretrained)
        self.criterion = nn.CrossEntropyLoss()
        self.T = T

    def info_nce_loss(self, features):

        bs = features.shape[0] // 2
        labels = torch.cat([torch.arange(bs) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda()
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.T
        loss = self.criterion(logits, labels)
        return loss

    def forward(self, img1, img2=None, return_loss=False):
        if not return_loss:
            return self.encoder(img1)[1]
        
        proj1, feat1 = self.encoder(img1)
        proj2, feat2 = self.encoder(img2)
        
        feats = torch.cat([feat1, feat2], dim=0)
        loss = self.info_nce_loss(feats)
        
        return feat1, feat2, loss