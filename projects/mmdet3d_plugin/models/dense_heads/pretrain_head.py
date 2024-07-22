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

from .nerf_utils import (visualize_image_semantic_depth_pair, 
                         visualize_image_pairs)
from .. import utils


OCC3D_PALETTE = torch.Tensor([
    [0, 0, 0],
    [255, 120, 50],  # barrier              orangey
    [255, 192, 203],  # bicycle              pink
    [255, 255, 0],  # bus                  yellow
    [0, 150, 245],  # car                  blue
    [0, 255, 255],  # construction_vehicle cyan
    [200, 180, 0],  # motorcycle           dark orange
    [255, 0, 0],  # pedestrian           red
    [255, 240, 150],  # traffic_cone         light yellow
    [135, 60, 0],  # trailer              brown
    [160, 32, 240],  # truck                purple
    [255, 0, 255],  # driveable_surface    dark pink
    [139, 137, 137], # other_flat           dark grey
    [75, 0, 75],  # sidewalk             dard purple
    [150, 240, 80],  # terrain              light green
    [230, 230, 250],  # manmade              white
    [0, 175, 0],  # vegetation           green
    [0, 255, 127],  # ego car              dark cyan
    [255, 99, 71],
    [0, 191, 255],
    [125, 125, 125]
])


@HEADS.register_module()
class PretrainHead(BaseModule):
    def __init__(
        self,
        in_channels=128,
        view_cfg=None,
        uni_conv_cfg=None,
        render_head_cfg=None,
        render_scale=(1, 1),
        use_semantic=False,
        semantic_class=17,
        vis_gt=False,
        vis_pred=False,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels

        self.use_semantic = use_semantic
        self.vis_gt = vis_gt
        self.vis_pred = vis_pred

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

        self.render_scale = render_scale

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
                img_depth,
                img_inputs,
                **kwargs):

        output = dict()

        # Prepare the projection parameters
        lidar2img, lidar2cam, intrinsics = [], [], []
        for img_meta in img_metas:
            lidar2img.append(img_meta["lidar2img"])
            lidar2cam.append(img_meta["lidar2cam"])
            intrinsics.append(img_meta["cam_intrinsic"])
        lidar2img = np.asarray(lidar2img)  # (bs, 6, 1, 4, 4)
        lidar2cam = np.asarray(lidar2cam)  # (bs, 6, 1, 4, 4)
        intrinsics = np.asarray(intrinsics)

        ref_tensor = img_feats[0].float()

        intrinsics = ref_tensor.new_tensor(intrinsics)
        pose_spatial = torch.inverse(
            ref_tensor.new_tensor(lidar2cam)
        )


        output['pose_spatial'] = pose_spatial[:, :, 0]
        output['intrinsics'] = intrinsics[:, :, 0]  # (bs, 6, 4, 4)
        output['intrinsics'][:, :, 0] *= self.render_scale[1]
        output['intrinsics'][:, :, 1] *= self.render_scale[0]

        if self.vis_gt:
            ## NOTE: due to the occ gt is labelled in the ego coordinate, we need to
            # use the cam2ego matrix as ego matrix
            cam2camego = []
            for img_meta in img_metas:
                cam2camego.append(img_meta["cam2camego"])
            cam2camego = np.asarray(cam2camego)  # (bs, 6, 1, 4, 4)
            output['pose_spatial'] = ref_tensor.new_tensor(cam2camego)

            gt_data_dict = self.prepare_gt_data(**kwargs)
            output.update(gt_data_dict)
            render_results = self.render_head(output, vis_gt=True)
            
            ## visualiza the results
            render_depth, rgb_pred, semantic_pred = render_results
            # current_frame_img = torch.zeros_like(rgb_pred).cpu().numpy()
            current_frame_img = img_inputs.cpu().numpy()
            visualize_image_semantic_depth_pair(
                current_frame_img[0],
                rgb_pred[0].permute(0, 2, 3, 1),
                render_depth[0],
                save_dir="results/vis/3dgs_baseline_gt"
            )
            exit()

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
        _uni_feats = rearrange(uni_feats, 'b c z y x -> b x y z c')

        output['volume_feat'] = _uni_feats

        occupancy_output = self.occupancy_head(_uni_feats)
        occupancy_output = rearrange(occupancy_output, 'b x y z dim1 -> b dim1 x y z')
        output['density_prob'] = occupancy_output  # density score

        semantic_output = self.semantic_head(_uni_feats) if self.use_semantic else None
        semantic_output = rearrange(semantic_output, 'b x y z C -> b C x y z')
        output['semantic'] = semantic_output

        ## 2. Start rendering, including neural rendering or 3DGS
        render_results = self.render_head(output)

        ## Visualize the results
        if self.vis_pred:
            render_depth = render_results['render_depth']
            rgb_pred = render_results['render_rgb'] * 255.0
            semantic_pred = render_results['render_semantic']

            render_gt_semantic = kwargs.get('render_gt_semantic', None)
            render_gt_depth = kwargs.get('render_gt_depth', None)

            semantic = semantic_pred.argmax(2)
            semantic = OCC3D_PALETTE[semantic].to(semantic_pred)
            visualize_image_pairs(
                img_inputs[0],
                semantic[0], # rgb_pred[0].permute(0, 2, 3, 1),
                render_depth[0],
                semantic_is_sparse=False,
                depth_is_sparse=False,
                save_dir="results/vis/3dgs_overfitting"
            )
            exit()
        return render_results
    
    def loss(self, preds_dict, targets):
        loss_dict = self.render_head.loss(preds_dict, targets)
        return loss_dict
    
    def prepare_gt_data(self, **kwargs):
        # Prepare the ground truth volume data for visualization
        voxel_semantics = kwargs['voxel_semantics']
        density_prob = rearrange(voxel_semantics, 'b x y z -> b () x y z')
        density_prob = density_prob != 17
        density_prob = density_prob.float()
        density_prob[density_prob == 0] = -10  # scaling to avoid 0 in alphas
        density_prob[density_prob == 1] = 10

        output = dict()
        output['density_prob'] = density_prob

        semantic = OCC3D_PALETTE[voxel_semantics.long()].to(density_prob)
        semantic = semantic.permute(0, 4, 1, 2, 3)  # to (b, 3, 200, 200, 16)
        output['semantic'] = semantic
        return output
        