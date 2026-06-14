import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1

class FacenetBaseline(nn.Module):
    def __init__(self, pretrained: str = 'vggface2'):
        super(FacenetBaseline, self).__init__()
        self.backbone = InceptionResnetV1(pretrained=pretrained)
        
    def forward(self, x: torch.Tensor):
        return self.backbone(x)
