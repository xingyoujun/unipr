import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from mmcv.runner import BaseModule
from mmcv.cnn import Linear, ConvModule, build_activation_layer
from mmdet.models.builder import BACKBONES

from .dinov2 import vit_base, vit_base_reg

    
@BACKBONES.register_module()
class DINOV2(BaseModule):
    def __init__(self, arch='base', load_path='dinov2_vits14_pretrain.pth',out_indices=[2,5,8,11], out_channels=[96, 192, 384, 768]):
        super().__init__()

        self.model = vit_base_reg(patch_size=14, img_size=518, init_values=1.0, block_chunks=0, interpolate_antialias=True, interpolate_offset=0.0)
        self.pretrained_dinov2 = load_path
        self.out_indices = out_indices
        self.readout_type = "project"

        in_channels=768
        self.out_channels = out_channels

        self.projects = nn.ModuleList(
            [
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    act_cfg=None,
                )
                for out_channel in out_channels
            ]
        )
        if len(self.out_indices) == 2:
            self.resize_layers = nn.ModuleList(
                [
                    nn.Identity(),
                    nn.Conv2d(
                        in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=3, stride=2, padding=1
                    ),
                ]
            )
        else:
            self.resize_layers = nn.ModuleList(
                [
                    nn.ConvTranspose2d(
                        in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                    ),
                    nn.ConvTranspose2d(
                        in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                    ),
                    nn.Identity(),
                    nn.Conv2d(
                        in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                    ),
                ]
            )

        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(Linear(2 * in_channels, in_channels), build_activation_layer(dict(type="GELU")))
                )

        self.init_weights()

    def init_weights(self):
        # pretrain_dict = torch.load(self.pretrained_dinov2, map_location='cpu')
        loaded_npz = np.load(self.pretrained_dinov2)
        loaded_state_dict = {key: torch.from_numpy(value) for key, value in loaded_npz.items()}
        msg = self.model.load_state_dict(loaded_state_dict, strict=True)
        self._freeze_stages()
        # for dinov2 init nothing
        return 

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = self.sam_model.image_encoder.img_size - h
        padw = self.sam_model.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def forward(self, inputs):
        inter = self.model.get_intermediate_layers(inputs, n=self.out_indices,reshape=True,return_class_token=True,norm=False)
        inter = [inp for inp in inter]
        assert isinstance(inter, list)
        out = []
        for i, x in enumerate(inter):
            assert len(x) == 2
            x, cls_token = x[0], x[1]
            feature_shape = x.shape
            if self.readout_type == "project":
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == "add":
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)
            else:
                pass
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out

    def _freeze_stages(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        super(DINOV2, self).train(mode)
        self._freeze_stages()
