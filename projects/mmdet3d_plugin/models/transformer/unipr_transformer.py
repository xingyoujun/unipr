import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from .decoder import CustomMSDeformableAttention
from .unipr_self_attention import UniprSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D

@TRANSFORMER.register_module()
class UniprTransformer(BaseModule):
    """Implements the UniprTransformer transformer.
    """

    def __init__(self,
                 num_feature_levels=1,
                 num_cams=2,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(UniprTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the VFTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, UniprSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @auto_fp16(apply_to=('mlvl_feats', 'tpv_queries', 'prev_tpv', 'tpv_pos'))
    def get_tpv_features(
            self,
            mlvl_feats,
            tpv_queries,
            tpv_u,
            tpv_v,
            tpv_d,
            tpv_pos=None,
            prev_tpv=None,
            **kwargs):
        """
        obtain tpv features.
        """
        bs = mlvl_feats[0].size(0)
        device = mlvl_feats[0].device

        if prev_tpv is not None:
            if prev_tpv.shape[1] == tpv_u * tpv_v * tpv_d:
                prev_tpv = prev_tpv.permute(1, 0, 2)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)

        tpv_embed = self.encoder(
            tpv_queries,
            feat_flatten,
            feat_flatten,
            tpv_u=tpv_u,
            tpv_v=tpv_v,
            tpv_d=tpv_d,
            tpv_pos=tpv_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_tpv=prev_tpv,
            **kwargs
        )
        
        return tpv_embed

    @auto_fp16(apply_to=('mlvl_feats', 'tpv_queries', 'object_query_embed', 'prev_tpv', 'tpv_pos'))
    def forward(self,
                mlvl_feats,
                tpv_queries,
                object_query_embeds,
                tpv_u,
                tpv_v,
                tpv_d,
                tpv_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_tpv=None,
                ref_point=None,
                **kwargs):
        """Forward function for `UniprTransformer`.
        """

        tpv_embed = self.get_tpv_features(
            mlvl_feats,
            tpv_queries,
            tpv_u,
            tpv_v,
            tpv_d,
            tpv_pos=tpv_pos,
            prev_tpv=prev_tpv,
            **kwargs)  # tpv_embed shape: bs, tpv_h*tpv_w, embed_dims

        bs = mlvl_feats[0].size(0)

        query_pos = object_query_embeds.unsqueeze(0).expand(bs, -1, -1) # bs nq c
        query = torch.zeros_like(query_pos)
        reference_points = ref_point
        init_reference_out = reference_points # bs nq 3

        spatial_shapes = torch.tensor([
            [tpv_v, tpv_u],
            [tpv_d, tpv_u],
            [tpv_d, tpv_v]
        ], device=query.device)
        level_start_index = torch.tensor([
            0, tpv_v*tpv_u, tpv_v*tpv_u+tpv_d*tpv_u
        ], device=query.device)
        
        tpv_embed = torch.cat(tpv_embed, dim=1)

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        tpv_embed = tpv_embed.permute(1, 0, 2)
        
        reference_points_uv = reference_points[...,[0,1]].clone()
        reference_points_ud = reference_points[...,[0,2]].clone()
        reference_points_vd = reference_points[...,[1,2]].clone()
        reference_points_tpv = torch.stack((reference_points_uv,reference_points_ud,reference_points_vd), dim=2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=tpv_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reference_points_cam=reference_points_tpv,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            **kwargs)
    
        inter_references_out = inter_references

        return tpv_embed, inter_states, init_reference_out, inter_references_out