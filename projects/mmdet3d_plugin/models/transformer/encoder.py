import numpy as np
import torch
import copy
import warnings

import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class UniprTransformerEncoder(TransformerLayerSequence):
    def __init__(self, *args, pc_range=None, tpv_u=30, tpv_v=30, tpv_d=128, num_points_in_pillar=[32,24,32], num_points_in_pillar_cross_view=[8,8,8], return_intermediate=False, **kwargs):

        super(UniprTransformerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.num_points_in_pillar_cross_view = num_points_in_pillar_cross_view
        self.pc_range = pc_range

        self.tpv_u = tpv_u
        self.tpv_v = tpv_v
        self.tpv_d = tpv_d

        ref_3d_uv = self.get_reference_points(tpv_v, tpv_u, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar[0], '3d', device='cpu')
        ref_3d_ud = self.get_reference_points(tpv_d, tpv_u, self.pc_range[4]-self.pc_range[1], self.num_points_in_pillar[1], '3d', device='cpu')
        ref_3d_vd = self.get_reference_points(tpv_d, tpv_v, self.pc_range[3]-self.pc_range[0], self.num_points_in_pillar[2], '3d', device='cpu')

        ref_3d_ud = ref_3d_ud[...,[0,2,1]]
        ref_3d_vd = ref_3d_vd[...,[2,0,1]]
        self.register_buffer('ref_3d_uv', ref_3d_uv)
        self.register_buffer('ref_3d_ud', ref_3d_ud)
        self.register_buffer('ref_3d_vd', ref_3d_vd)

        cross_view_ref_points = self.get_cross_view_ref_points(tpv_d, tpv_v, tpv_u, num_points_in_pillar_cross_view)
        self.register_buffer('cross_view_ref_points', cross_view_ref_points)

        self.fp16_enabled = False

    @staticmethod
    def get_cross_view_ref_points(tpv_d, tpv_v, tpv_u, num_points_in_pillar):
        # ref points generating target: (#query)hw+zh+wz, (#level)3, #p, 2
        # generate points for hw and level 1
        v_ranges = torch.linspace(0.5, tpv_v-0.5, tpv_v) / tpv_v
        u_ranges = torch.linspace(0.5, tpv_u-0.5, tpv_u) / tpv_u
        v_ranges = v_ranges.unsqueeze(-1).expand(-1, tpv_u).flatten()
        u_ranges = u_ranges.unsqueeze(0).expand(tpv_v, -1).flatten()
        uv_uv = torch.stack([u_ranges, v_ranges], dim=-1) # hw, 2
        uv_uv = uv_uv.unsqueeze(1).expand(-1, num_points_in_pillar[2], -1) # hw, #p, 2
        # generate points for hw and level 2
        d_ranges = torch.linspace(0.5, tpv_d-0.5, num_points_in_pillar[2]) / tpv_d # #p
        d_ranges = d_ranges.unsqueeze(0).expand(tpv_v*tpv_u, -1) # hw, #p
        v_ranges = torch.linspace(0.5, tpv_v-0.5, tpv_v) / tpv_v
        v_ranges = v_ranges.reshape(-1, 1, 1).expand(-1, tpv_u, num_points_in_pillar[2]).flatten(0, 1)
        uv_vd = torch.stack([v_ranges, d_ranges], dim=-1) # hw, #p, 2
        # generate points for hw and level 3
        d_ranges = torch.linspace(0.5, tpv_d-0.5, num_points_in_pillar[2]) / tpv_d # #p
        d_ranges = d_ranges.unsqueeze(0).expand(tpv_v*tpv_u, -1) # hw, #p
        u_ranges = torch.linspace(0.5, tpv_u-0.5, tpv_u) / tpv_u
        u_ranges = u_ranges.reshape(1, -1, 1).expand(tpv_v, -1, num_points_in_pillar[2]).flatten(0, 1)
        uv_ud = torch.stack([u_ranges, d_ranges], dim=-1) # hw, #p, 2
        
        # generate points for zh and level 1
        u_ranges = torch.linspace(0.5, tpv_u-0.5, num_points_in_pillar[1]) / tpv_u
        u_ranges = u_ranges.unsqueeze(0).expand(tpv_d*tpv_v, -1)
        v_ranges = torch.linspace(0.5, tpv_v-0.5, tpv_v) / tpv_v
        v_ranges = v_ranges.reshape(1, -1, 1).expand(tpv_d, -1, num_points_in_pillar[1]).flatten(0, 1)
        vd_uv = torch.stack([u_ranges, v_ranges], dim=-1)
        # generate points for zh and level 2
        d_ranges = torch.linspace(0.5, tpv_d-0.5, tpv_d) / tpv_d
        d_ranges = d_ranges.reshape(-1, 1, 1).expand(-1, tpv_v, num_points_in_pillar[1]).flatten(0, 1)
        v_ranges = torch.linspace(0.5, tpv_v-0.5, tpv_v) / tpv_v
        v_ranges = v_ranges.reshape(1, -1, 1).expand(tpv_d, -1, num_points_in_pillar[1]).flatten(0, 1)
        vd_vd = torch.stack([v_ranges, d_ranges], dim=-1) # zh, #p, 2
        # generate points for zh and level 3
        u_ranges = torch.linspace(0.5, tpv_u-0.5, num_points_in_pillar[1]) / tpv_u
        u_ranges = u_ranges.unsqueeze(0).expand(tpv_d*tpv_v, -1)
        d_ranges = torch.linspace(0.5, tpv_d-0.5, tpv_d) / tpv_d
        d_ranges = d_ranges.reshape(-1, 1, 1).expand(-1, tpv_v, num_points_in_pillar[1]).flatten(0, 1)
        vd_ud = torch.stack([u_ranges, d_ranges], dim=-1)

        # generate points for wz and level 1
        v_ranges = torch.linspace(0.5, tpv_v-0.5, num_points_in_pillar[0]) / tpv_v
        v_ranges = v_ranges.unsqueeze(0).expand(tpv_d*tpv_u, -1)
        u_ranges = torch.linspace(0.5, tpv_u-0.5, tpv_u) / tpv_u
        u_ranges = u_ranges.reshape(1, -1, 1).expand(tpv_d, -1, num_points_in_pillar[0]).flatten(0, 1)
        ud_uv = torch.stack([u_ranges, v_ranges], dim=-1)
        # generate points for wz and level 2
        v_ranges = torch.linspace(0.5, tpv_v-0.5, num_points_in_pillar[0]) / tpv_v
        v_ranges = v_ranges.unsqueeze(0).expand(tpv_d*tpv_u, -1)
        d_ranges = torch.linspace(0.5, tpv_d-0.5, tpv_d) / tpv_d
        d_ranges = d_ranges.reshape(-1, 1, 1).expand(-1, tpv_u, num_points_in_pillar[0]).flatten(0, 1)
        ud_vd = torch.stack([v_ranges, d_ranges], dim=-1)
        # generate points for wz and level 3
        u_ranges = torch.linspace(0.5, tpv_u-0.5, tpv_u) / tpv_u
        u_ranges = u_ranges.reshape(1, -1, 1).expand(tpv_d, -1, num_points_in_pillar[0]).flatten(0, 1)
        d_ranges = torch.linspace(0.5, tpv_d-0.5, tpv_d) / tpv_d
        d_ranges = d_ranges.reshape(-1, 1, 1).expand(-1, tpv_u, num_points_in_pillar[0]).flatten(0, 1)
        ud_ud = torch.stack([u_ranges, d_ranges], dim=-1)

        reference_points = torch.cat([
            torch.stack([uv_uv, uv_vd, uv_ud], dim=1),
            torch.stack([ud_uv, ud_vd, ud_ud], dim=1),
            torch.stack([vd_uv, vd_vd, vd_ud], dim=1),
        ], dim=0) # uv+ud+vd, 3, #p, 2
        
        return reference_points

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D tpv plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    # This function must use fp32!!!
    @force_fp32(apply_to=('reference_points', 'img_metas'))
    def point_sampling(self, reference_points, pc_range,  img_metas):
        # NOTE: close tf32 here.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]

        reference_points_cam_left = reference_points[...,:2].clone()
        reference_points_cam_right = reference_points.clone()
        reference_points_cam_right[...,0:1] *= pad_w
        # as depth grow bf low
        reference_points_cam_right[..., 2:3] = reference_points_cam_right[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
        reference_points_cam_right[...,0:1] -= reference_points_cam_right[..., 2:3]
        reference_points_cam_right[...,0:1] /= pad_w
        reference_points_cam_right = reference_points_cam_right[...,:2]
        
        reference_points_cam = torch.stack((reference_points_cam_left, reference_points_cam_right), 1)
        tpv_mask = ((reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0))

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            tpv_mask = torch.nan_to_num(tpv_mask)
        else:
            tpv_mask = tpv_mask.new_tensor(
                np.nan_to_num(tpv_mask.cpu().numpy()))

        reference_points_cam = reference_points_cam.permute(1, 0, 3, 2, 4)
        tpv_mask = tpv_mask.permute(1, 0, 3, 2, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, tpv_mask
        
    def forward(self,
                tpv_query,
                key,
                value,
                tpv_u=None,
                tpv_v=None,
                tpv_d=None,
                tpv_pos=None,
                spatial_shapes=None,
                level_end_index=None,
                prev_tpv=None,
                **kwargs):
        """Forward function for `UniprTransformerEncoder`.
        """
        output = tpv_query
        intermediate = []
        bs = tpv_query[0].shape[0]

        reference_points_cams, tpv_masks = [], []
        ref_3d_uv = self.ref_3d_uv.repeat(bs, 1, 1, 1)
        ref_3d_ud = self.ref_3d_ud.repeat(bs, 1, 1, 1)
        ref_3d_vd = self.ref_3d_vd.repeat(bs, 1, 1, 1)

        ref_3ds = [ref_3d_uv, ref_3d_ud, ref_3d_vd]
        for ref_3d in ref_3ds:
            reference_points_cam, tpv_mask = self.point_sampling(
                ref_3d, self.pc_range, kwargs['img_metas']) # num_cam, bs, hw++, #p, 2
            reference_points_cams.append(reference_points_cam)
            tpv_masks.append(tpv_mask)

        ref_cross_view = self.cross_view_ref_points.clone().unsqueeze(0).expand(bs, -1, -1, -1, -1)

        for lid, layer in enumerate(self.layers):
            output = layer(
                tpv_query,
                key,
                value,
                tpv_pos=tpv_pos,
                ref_2d=ref_cross_view,
                ref_3d=ref_3ds,
                tpv_u=tpv_u,
                tpv_v=tpv_v,
                tpv_d=tpv_d,
                spatial_shapes=spatial_shapes,
                level_end_index=level_end_index,
                reference_points_cam=reference_points_cams,
                tpv_mask=tpv_masks,
                prev_tpv=prev_tpv,
                **kwargs)

            tpv_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


@TRANSFORMER_LAYER.register_module()
class UniprTransformerEncoderLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default: None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default: 2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(UniprTransformerEncoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False

        assert len(operation_order) == 6

    def forward(self,
                query,
                key,
                value,
                tpv_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                tpv_u=None,
                tpv_v=None,
                tpv_d=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                reference_points_cam=None,
                tpv_mask=None,
                prev_tpv=None,
                **kwargs):
        """Forward function for `UniprTransformerEncoderLayer`.
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        if self.operation_order[0] == 'cross_attn':
            query = torch.cat(query, dim=1)
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                
                self_spatial_shapes = torch.tensor([
                    [tpv_v, tpv_u],
                    [tpv_d, tpv_u],
                    [tpv_d, tpv_v]
                ], device=query[0].device)
                self_level_start_index = torch.tensor([
                    0, tpv_v*tpv_u, tpv_v*tpv_u+tpv_d*tpv_u
                ], device=query[0].device)

                if not isinstance(query, (list, tuple)):
                    query = torch.split(query, [tpv_v*tpv_u, tpv_d*tpv_u, tpv_d*tpv_v], dim=1)

                query = self.attentions[attn_index](
                    query,
                    prev_tpv,
                    prev_tpv,
                    identity if self.pre_norm else None,
                    query_pos=tpv_pos,
                    key_pos=tpv_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    spatial_shapes=self_spatial_shapes,
                    level_start_index=self_level_start_index,
                    **kwargs)
                attn_index += 1
                query = torch.cat(query, dim=1)
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=tpv_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cams=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    tpv_masks=tpv_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1
        query = torch.split(query, [tpv_v*tpv_u, tpv_d*tpv_u, tpv_d*tpv_v], dim=1)
        return query