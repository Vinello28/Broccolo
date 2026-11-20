import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VisionEncoder(nn.Module):
    def __init__(self, model_name='vit_b_16', pretrained=True):
        super(VisionEncoder, self).__init__()
        if model_name == 'vit_b_16':
            self.model = models.vit_b_16(pretrained=pretrained)
            self.feature_dim = self.model.heads.head.in_features
            self.model.heads.head = nn.Identity()  # Remove the classification head
        else:
            raise ValueError(f"Model {model_name} not supported.")

    def forward(self, x):
        features = self.model(x)
        return features