import torch
import torch.nn as nn
import torchvision.models as models


class FightDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(FightDetector, self).__init__()

        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):

        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        out = self.backbone(x)

        out = out.view(b, t, -1)
        out = torch.mean(out, dim=1)

        return out
