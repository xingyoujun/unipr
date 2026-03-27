import torch
import torch.nn.functional as F
from mmdet.core.bbox.match_costs.builder import MATCH_COST

@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class Rot3dCost(object):
    """Rot3dCost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, pred_rots, gt_rots):
        """
        Args:
            pred_rots (Tensor): 9d matrixs.
            gt_bboxes (Tensor): 9d matrixs.
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        n_q = pred_rots.shape[0]
        n_g = gt_rots.shape[0]
        
        p_x = pred_rots[:,0:3].view(-1, 3, 1)
        p_y = pred_rots[:,3:6].view(-1, 3, 1)
        p_z = pred_rots[:,6:9].view(-1, 3, 1)
        p_matrix = torch.cat((p_x, p_y, p_z), 2) #n_q,3,3
        p_matrix = p_matrix.unsqueeze(1).repeat(1,n_g,1,1).view(-1,3,3)

        g_x = gt_rots[:,0:3].view(-1, 3, 1)
        g_y = gt_rots[:,3:6].view(-1, 3, 1)
        g_z = gt_rots[:,6:9].view(-1, 3, 1)
        g_matrix = torch.cat((g_x, g_y, g_z), 2) #n_q,3,3
        g_matrix = g_matrix.unsqueeze(0).repeat(n_q,1,1,1).view(-1,3,3)
        
        m = torch.bmm(p_matrix, g_matrix.transpose(1, 2))  # b*3*3
        m_trace = torch.einsum("bii->b", m)  # batch trace

        cos = (m_trace - 1) / 2  # [-1, 1]
        eps = 1e-6
        cos = torch.clamp(cos, -1+eps, 1-eps)  # avoid nan
        dist = (1 - cos) / 2  # [0, 1]
        rot_cost = dist.view(n_q, n_g)
    
        return rot_cost * self.weight
    
@MATCH_COST.register_module()
class SigmoidCeLCost(object):
    """SigmoidCeLCost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        cls_pred = cls_pred.flatten(1).float()
        if len(gt_labels.shape) == 1:
            gt_labels = gt_labels.view(-1, 1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        # _logits no need sigmoid
        # TODO for now 0 is pos 1 is neg
        pos = F.binary_cross_entropy_with_logits(
            cls_pred, torch.zeros_like(cls_pred), reduction='none')
        neg = F.binary_cross_entropy_with_logits(
            cls_pred, torch.ones_like(cls_pred), reduction='none')
        cls_cost = torch.einsum('nc,mc->nm', pos, gt_labels) + \
            torch.einsum('nc,mc->nm', neg, 1 - gt_labels)
        cls_cost = cls_cost / n
        
        return self.weight * cls_cost