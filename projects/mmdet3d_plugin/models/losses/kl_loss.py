import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class KLLoss(nn.Module):
    def __init__(self,  loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = 'mean'

    def forward(self, inputs, targets, avg_factor):
        """
        Forward function to calculate accuracy.
        """  
        if targets.numel() == 0:
            return inputs.sum() * 0

        assert inputs.size() == targets.size()

        input_mean = inputs[:,:64]
        input_log_var = inputs[:,64:]
        input_var = torch.exp(input_log_var)
        
        gt_mean = targets[:,:64]
        gt_log_var = targets[:,64:]
        gt_var = torch.exp(gt_log_var)

        assert self.reduction == "mean"
        loss = 0.5 * torch.mean(torch.pow(input_mean - gt_mean, 2) / gt_var + input_var / gt_var - 1.0 - input_log_var + gt_log_var)

        return self.loss_weight * loss