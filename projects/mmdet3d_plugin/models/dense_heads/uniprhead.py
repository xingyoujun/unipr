import copy
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

from projects.mmdet3d_plugin.models.utils import kl_d256_m512_l64, DiagonalGaussianDistribution

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

@HEADS.register_module()
class UniprHead(DETRHead):
    def __init__(self,
                 *args,
                 load_ae_path=None,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 tpv_u=30,
                 tpv_v=30,
                 tpv_d=128,
                 w_scale=False,
                 **kwargs):
        
        self.tpv_u = tpv_u
        self.tpv_v = tpv_v
        self.tpv_d = tpv_d
        self.fp16_enabled = False
        self.w_scale = w_scale

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 5 # default
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.real_d = self.pc_range[5] - self.pc_range[2]
        self.num_cls_fcs = num_cls_fcs - 1

        self.load_ae_path = load_ae_path

        super(UniprHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)
        self.loss_embedding = build_loss(kwargs['loss_embedding'])
        self.loss_voxel = torch.nn.BCEWithLogitsLoss()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        mean_embedding_branch = []
        for _ in range(3):
            mean_embedding_branch.append(Linear(self.embed_dims, self.embed_dims))
            mean_embedding_branch.append(nn.ReLU())
        mean_embedding_branch.append(Linear(self.embed_dims, 64)) # for mean and logvar
        mean_embedding_branch = nn.Sequential(*mean_embedding_branch)

        logvar_embedding_branch = []
        for _ in range(3):
            logvar_embedding_branch.append(Linear(self.embed_dims, self.embed_dims))
            logvar_embedding_branch.append(nn.ReLU())
        logvar_embedding_branch.append(Linear(self.embed_dims, 64)) # for mean and logvar
        logvar_embedding_branch = nn.Sequential(*logvar_embedding_branch)

        self.num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])
        self.mean_embedding_branches = nn.ModuleList(
            [mean_embedding_branch for _ in range(self.num_pred)])
        self.logvar_embedding_branches = nn.ModuleList(
            [logvar_embedding_branch for _ in range(self.num_pred)])

        if not self.as_two_stage:
            self.tpv_embedding_uv = nn.Embedding(self.tpv_v * self.tpv_u, self.embed_dims)
            self.tpv_embedding_ud = nn.Embedding(self.tpv_d * self.tpv_u, self.embed_dims)
            self.tpv_embedding_vd = nn.Embedding(self.tpv_d * self.tpv_v, self.embed_dims)

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(128*3, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.ae_model = kl_d256_m512_l64()
        self.ae_model.load_state_dict(torch.load(self.load_ae_path, map_location='cpu')['model'], strict=True)

        for param in self.ae_model.parameters():
            param.requires_grad = False

    def init_weights(self):
        """Initialize weights of the transformer head."""
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, prev_tpv=None,  only_tpv=False):
        """Forward function.
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        device = mlvl_feats[0].device

        reference_points = self.reference_points.weight
        object_query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1) #.sigmoid()

        tpv_queries_uv = self.tpv_embedding_uv.weight.to(dtype)
        tpv_queries_ud = self.tpv_embedding_ud.weight.to(dtype)
        tpv_queries_vd = self.tpv_embedding_vd.weight.to(dtype)

        tpv_queries_uv = tpv_queries_uv.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_ud = tpv_queries_ud.unsqueeze(0).repeat(bs, 1, 1)
        tpv_queries_vd = tpv_queries_vd.unsqueeze(0).repeat(bs, 1, 1)

        tpv_pos_uv = self.positional_encoding(bs, device, 'z')
        tpv_pos_ud = self.positional_encoding(bs, device, 'h')
        tpv_pos_vd = self.positional_encoding(bs, device, 'w')
        tpv_pos = [tpv_pos_uv, tpv_pos_ud, tpv_pos_vd]

        if only_tpv:
            return self.transformer.get_tpv_features(
                mlvl_feats,
                [tpv_queries_uv, tpv_queries_ud, tpv_queries_vd],
                self.tpv_u,
                self.tpv_v,
                self.tpv_d,
                tpv_pos=tpv_pos,
                img_metas=img_metas,
                prev_tpv=prev_tpv,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                [tpv_queries_uv, tpv_queries_ud, tpv_queries_vd],
                object_query_embeds,
                self.tpv_u,
                self.tpv_v,
                self.tpv_d,
                tpv_pos=tpv_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_tpv=prev_tpv,
                ref_point = reference_points,
        )

        tpv_embed, hs, init_reference, inter_references = outputs
        
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_embeddings = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            outputs_mean_embedding = self.mean_embedding_branches[lvl](hs[lvl])
            outputs_logvar_embedding = self.logvar_embedding_branches[lvl](hs[lvl])

            outputs_embedding = torch.cat((outputs_mean_embedding, outputs_logvar_embedding), dim=-1)
            assert reference.shape[-1] == 3
            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_embeddings.append(outputs_embedding)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_embedding_preds = torch.stack(outputs_embeddings)

        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1])
        all_bbox_preds[..., 2:3] = (all_bbox_preds[..., 2:3] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2])

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'all_embedding_preds': all_embedding_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        gt_pos_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_bboxes.new_full((num_bboxes, ), cls_score.shape[-1], dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = bbox_pred.new_zeros(bbox_pred.shape[0], code_size)
        bbox_weights = bbox_pred.new_zeros(bbox_pred.shape[0], code_size)
        bbox_weights[pos_inds] = 1.0
        # TODO just work BUG from https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/samplers/sampling_result.py#L51
        if len(sampling_result.pos_gt_bboxes) == 0:
            sampling_result.pos_gt_bboxes = sampling_result.pos_gt_bboxes.view(-1, bbox_targets.shape[-1])
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds, gt_pos_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, pos_inds_list, neg_inds_list, gt_pos_inds_list = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list, gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, pos_inds_list, neg_inds_list, gt_pos_inds_list)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    embedding_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_embeddings_list,
                    gt_voxels_points_list,
                    gt_voxels_label_list,
                    gt_bboxes_ignore_list=None,
                    img_metas=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        if cls_scores.shape[1] == 0:
            assert self.sync_cls_avg_factor==False
            num_total_pos = cls_scores.new_tensor([1])
            num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
            return cls_scores.sum(), bbox_preds.sum(), None, embedding_preds.sum()

        num_imgs, num_q = cls_scores.shape[:2]
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list, num_total_pos, num_total_neg, pos_inds_list, _, gt_pos_inds_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        for bs_id in range(len(pos_inds_list)):
            pos_inds_list[bs_id] += num_q * bs_id
        
        pos_inds = torch.cat(pos_inds_list, 0)
        
        gt_embeddings = []
        for bb in range(len(gt_pos_inds_list)):
            gt_pos_inds = gt_pos_inds_list[bb]
            gt_embedding = gt_embeddings_list[bb][gt_pos_inds]
            gt_embeddings.append(gt_embedding)
        gt_embeddings = torch.cat(gt_embeddings, 0)

        gt_voxels_points = []
        for bb in range(len(gt_pos_inds_list)):
            gt_pos_inds = gt_pos_inds_list[bb]
            gt_voxels_point = gt_voxels_points_list[bb][gt_pos_inds]
            gt_voxels_points.append(gt_voxels_point)
        gt_voxels_points = torch.cat(gt_voxels_points, 0)

        gt_voxels_labels = []
        for bb in range(len(gt_pos_inds_list)):
            gt_pos_inds = gt_pos_inds_list[bb]
            gt_voxels_label = gt_voxels_label_list[bb][gt_pos_inds]
            gt_voxels_labels.append(gt_voxels_label)
        gt_voxels_labels = torch.cat(gt_voxels_labels, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, cls_scores.shape[-1])
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)

        loss_bbox = self.loss_bbox(bbox_preds[pos_inds, :4], normalized_bbox_targets[pos_inds, :4], bbox_weights[pos_inds, :4], avg_factor=num_total_pos)
        
        embedding_preds = embedding_preds.reshape(-1, embedding_preds.size(-1))
        loss_embedding = self.loss_embedding(embedding_preds[pos_inds], gt_embeddings, avg_factor=num_total_pos)

        embedding_matches = embedding_preds[pos_inds]
        mean = embedding_matches[:,:64].reshape(-1, 1, 64)
        logvar = embedding_matches[:,64:].reshape(-1, 1, 64)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        sampled_embedding = posterior.sample()
    
        voxel_output = self.ae_model.decode_emb(sampled_embedding, gt_voxels_points)['logits']
        loss_voxel = self.loss_voxel(voxel_output, gt_voxels_labels) * 20.0

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        loss_embedding = torch.nan_to_num(loss_embedding)
        loss_voxel = torch.nan_to_num(loss_voxel)
        
        return loss_cls, loss_bbox, loss_embedding, loss_voxel
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             gt_parts_list,
             gt_embeddings_list,
             part_bboxes_list,
             part_labels_list,
             part_embeddings_list,
             gt_voxels_points,
             gt_voxels_label,
             gt_l_masks_list,
             gt_r_masks_list,
             gt_mask_list,
             gt_disp_list,
             preds_dicts,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        all_embedding_preds = preds_dicts['all_embedding_preds']
        
        num_dec_layers = len(all_cls_scores)
        
        device = gt_labels_list[0].device
        gt_bboxes_list = [gt_bboxes.tensor.to(device) for gt_bboxes in gt_bboxes_list]

        # change to uvd
        for i in range(len(img_metas)):
            if len(gt_bboxes_list[0]) != 0:
                bf = img_metas[i]['intrinsics'][0][0][0] * img_metas[i]['interocular_distance']
                intrinsics = img_metas[i]['intrinsics'][0]
                intrinsics = np.asarray(intrinsics)
                intrinsics = all_cls_scores.new_tensor(intrinsics) # (B, N, 4, 4)
                xyz = gt_bboxes_list[i][:,:3]
                xyz = torch.cat((xyz,xyz.new_ones((xyz.shape[0],1))), dim=-1)
                uvd = torch.matmul(intrinsics, xyz.T).T
                uvd[:,:2] = uvd[:,:2] / uvd[:,2:3]
                uvd[...,2] = bf / uvd[...,2]
                gt_bboxes_list[i][:,:3] = uvd[:, :3]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_embeddings_list = [gt_embeddings_list for _ in range(num_dec_layers)]
        
        gt_voxels_points = [gt_voxels_points for _ in range(num_dec_layers)]
        gt_voxels_label = [gt_voxels_label for _ in range(num_dec_layers)]

        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_embedding, losses_voxel = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, all_embedding_preds,
            all_gt_bboxes_list, all_gt_labels_list, all_gt_embeddings_list, gt_voxels_points, gt_voxels_label, all_gt_bboxes_ignore_list, img_metas=img_metas)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        loss_dict['loss_embedding'] = losses_embedding[-1]
        loss_dict['loss_voxel'] = losses_voxel[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_embedding_i, loss_voxel_i in zip(losses_cls[:-1],
                                                                    losses_bbox[:-1],
                                                                    losses_embedding[:-1],
                                                                    losses_voxel[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_embedding'] = loss_embedding_i
            loss_dict[f'd{num_dec_layer}.loss_voxel'] = loss_voxel_i
            num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            masks = preds['masks']
            embeddings = preds['embeddings']
            ret_list.append([bboxes, scores, labels, masks, embeddings])
        return ret_list
