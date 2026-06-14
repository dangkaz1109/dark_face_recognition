import torch
import torch.nn as nn

class StandardTripletLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super(StandardTripletLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return self.triplet_loss(anchor, positive, negative)

