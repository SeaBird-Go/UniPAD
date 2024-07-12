'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-07-10 11:08:49
Email: haimingzhang@link.cuhk.edu.cn
Description: The Pretraining head.
'''
import os
import os.path as osp
import numpy as np
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import HEADS
from mmdet3d.models import builder

from .. import utils


@HEADS.register_module()
class PretrainHead(BaseModule):
    def __init__(
        self,
        in_channels=128,
        view_cfg=None,
        uni_conv_cfg=None,
        render_head_cfg=None,
        use_semantic=False,
        semantic_class=17,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels

        self.use_semantic = use_semantic

        if view_cfg is not None:
            vtrans_type = view_cfg.pop('type', 'Uni3DViewTrans')
            self.view_trans = getattr(utils, vtrans_type)(**view_cfg)

        if uni_conv_cfg is not None:
            self.uni_conv = nn.Sequential(
                nn.Conv3d(
                    uni_conv_cfg["in_channels"],
                    uni_conv_cfg["out_channels"],
                    kernel_size=uni_conv_cfg["kernel_size"],
                    padding=uni_conv_cfg["padding"],
                    stride=1,
                ),
                nn.BatchNorm3d(uni_conv_cfg["out_channels"]),
                nn.ReLU(inplace=True),
            )

        if render_head_cfg is not None:
            self.render_head = builder.build_head(render_head_cfg)

        out_dim = uni_conv_cfg["out_channels"]
        if use_semantic:
            self.semantic_head = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Softplus(),
                nn.Linear(out_dim * 2, semantic_class),
            )

        self.occupancy_head = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Softplus(),
                nn.Linear(out_dim * 2, 1),
            )
        
    @auto_fp16(apply_to=("pts_feats", "img_feats", "img_depth"))
    def forward(self, 
                pts_feats, 
                img_feats, 
                img_metas, 
                img_depth):
        ## 1. Prepare the volume feature from the pts features and img features
        uni_feats = []
        if img_feats is not None:
            uni_feats.append(
                self.view_trans(img_feats, img_metas=img_metas, img_depth=img_depth)
            )
        if pts_feats is not None:
            uni_feats.append(pts_feats)

        uni_feats = sum(uni_feats)
        uni_feats = self.uni_conv(uni_feats)  # (bs, c, z, y, x)

        ## 2. Prepare the features for rendering
        output = dict()
        
        _uni_feats = rearrange(uni_feats, 'b c z y x -> b x y z c')

        output['volume_feat'] = _uni_feats

        occupancy_output = self.occupancy_head(_uni_feats)
        occupancy_output = rearrange(occupancy_output, 'b x y z dim1 -> b dim1 x y z')
        output['occupancy_score'] = occupancy_output  # density score

        semantic_output = self.semantic_head(_uni_feats) if self.use_semantic else None
        output['semantic'] = semantic_output

        ## 2. Start rendering, including neural rendering or 3DGS
        lidar2img, lidar2cam, intrinsics = [], [], []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            lidar2cam.append(img_meta["lidar2cam"])
            intrinsics.append(img_meta["cam_intrinsic"])
        lidar2img = np.asarray(lidar2img)  # (bs, 6, 1, 4, 4)
        lidar2cam = np.asarray(lidar2cam)  # (bs, 6, 1, 4, 4)
        intrinsics = np.asarray(intrinsics)

        intrinsics = uni_feats.new_tensor(intrinsics)
        pose_spatial = torch.inverse(
            uni_feats.new_tensor(lidar2cam)
        )

        output['pose_spatial'] = pose_spatial[:, :, 0]
        output['intrinsics'] = intrinsics[:, :, 0]

        render_results = self.render_head(output)
        return render_results
    
        