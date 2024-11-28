import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torch.autograd import Variable

class DSA_loss(nn.Module):
    """
    this loss function should support mse loss and infoNCE loss.
    """

    def __init__(self, loss_function, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.loss_function = loss_function  # -- default CrossEntropy
        self.device = device

        # choose loss function
        self.if_infoNCE = False

    def mse_loss(self, pred, target):
        """
        Args:
            pred (Tensor): NxC input features.
            target (Tensor): NxC target features.
        """
        N = pred.size(0)
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = 2 - 2 * (pred_norm * target_norm).sum() / N
        return loss

    def forward(self, image_features1, image_features2, logit_scale):
        if self.if_infoNCE is not True:
            b, c, n = image_features1.shape
            # feat1 = image_features1.transpose(2, 1).reshape(b * n, c)  #  这里对比原来的方法少了一个mlp映射，相当于少了一个特征空间对齐，交给backbone去做吧
            # feat2 = image_features2.transpose(2, 1).reshape(b * n, c)

            feat1 = image_features1.transpose(2, 1).reshape(b, c*n)  # 这里对比原来的方法少了一个mlp映射，相当于少了一个特征空间对齐，交给backbone去做吧
            feat2 = image_features2.transpose(2, 1).reshape(b, c*n)

            loss = self.mse_loss(feat1, feat2)

        else:
            # use infoNCE as loss
            b, c, n = image_features1.shape
            feat1 = image_features1.reshape(b, -1)
            feat2 = image_features2.reshape(b, -1)
            image_features1 = F.normalize(feat1, dim=-1)
            image_features2 = F.normalize(feat2, dim=-1)

            logits_per_image1 = logit_scale * image_features1 @ image_features2.T
            logits_per_image2 = logits_per_image1.T
            labels = torch.arange(len(logits_per_image1), dtype=torch.long, device=self.device)
            loss = (self.loss_function(logits_per_image1, labels) + self.loss_function(logits_per_image2, labels)) / 2

        return loss
