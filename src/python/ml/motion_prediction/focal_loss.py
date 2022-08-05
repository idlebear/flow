
# Focal Loss implementation for balencing uneven datasets.
#
# Implementation based on the paper:
#
#  Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.
#
# With reference to:
#   https://amaarora.gihub.io/2020/06/29/FocalLoss.html
import torch
import torch.nn as nn

# use Focal Loss (#ref) to alter the weighting and improve training
# alpha is factored based on whether this is a true instance (=1) or a true negative (=0)
#
class FocalLoss():
    def __init__(self, alpha, gamma, device='cpu'):
        self.alpha_t = torch.tensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    @staticmethod
    def LossFn(alpha=0.2, gamma=2.0, device='cpu'):
        loss = FocalLoss(alpha, gamma, device)
        return loss.forward

    def forward(self, prediction, target):
        loss = nn.functional.binary_cross_entropy_with_logits(prediction, target, reduction='none').view(-1)
        pt = torch.exp(-loss)
        target = target.type(torch.long)
        alpha_t = self.alpha_t[0] + (self.alpha_t[1] - self.alpha_t[0]) * target.reshape(-1)
        # alpha_t = torch.gather(self.alpha_t, 0, target.reshape(-1))
        return (alpha_t * (1 - pt) ** self.gamma * loss).mean()  # calculate the focal loss
