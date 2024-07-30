import torch
import torch.nn as nn
import torch.nn.functional as F

from models.registry import LOSSES
from tools.function import ratio2weight,ratio3weight


@LOSSES.register("bceloss")
class BCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None

    def forward(self, logits, targets):
        logits = logits[0]

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)

            loss_m = (loss_m * sample_weight.cuda())

        # losses = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()

        return [loss], [loss_m]

@LOSSES.register("focalloss")
class FocalLoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None,gamma=2, alpha=None):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.tb_writer = tb_writer

    def forward(self, logits, targets):
        logits = logits[0]

        # Compute the probability of each class
        probs = torch.sigmoid(logits)

        # Compute the focal loss
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            # Adjust the focal loss using the sample weights
            
            
            ratio = torch.from_numpy(self.sample_weight).type_as(targets_mask)
            # --------------------- AAAI ---------------------
            pos_weights = torch.sqrt(1 / (2 * ratio.sqrt())) * targets_mask
            neg_weights = torch.sqrt(1 / (2 * (1 - ratio.sqrt()))) * (1 - targets_mask)
            weights = pos_weights + neg_weights
            # sample_weight = ratio2weight(targets_mask, self.sample_weight)
            focal_loss = (focal_loss * weights.cuda())

        loss = focal_loss.sum(1).mean() if self.size_sum else focal_loss.sum()

        return [loss], [focal_loss]
