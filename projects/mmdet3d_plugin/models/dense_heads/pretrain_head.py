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
import random
import torch.nn.functional as F
from mmcv.runner.base_module import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import HEADS
from mmdet3d.models import builder

from .nerf_utils import (visualize_image_semantic_depth_pair, 
                         visualize_image_pairs)
from .. import utils
from .depth_ssl import *


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
        use_depth_consistency=False,
        render_view_indices=list(range(6)),
        depth_ssl_size=None,
        depth_loss_weight=1.0,
        opt=None,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels

        self.use_semantic = use_semantic
        self.vis_gt = vis_gt
        self.vis_pred = vis_pred

        ## use the depth self-supervised consistency loss
        self.use_depth_consistency = use_depth_consistency
        self.render_view_indices = render_view_indices
        self.depth_ssl_size = depth_ssl_size
        self.opt = opt  # options for the depth consistency loss
        self.depth_loss_weight = depth_loss_weight

        if self.use_depth_consistency:
            h = depth_ssl_size[0]
            w = depth_ssl_size[1]
            num_cam = len(self.render_view_indices)
            self.backproject_depth = BackprojectDepth(num_cam, h, w)
            self.project_3d = Project3D(num_cam, h, w)

            self.ssim = SSIM()

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
            from .nerf_utils import VisElement, visualize_elements

            save_dir = "results/vis/3dgs_overfitting_offset_scale_0.5_rescale_z_axis"
            os.makedirs(save_dir, exist_ok=True)

            ## save the occupancy offline for visualization
            torch.save(semantic_output.detach().cpu(), f'{save_dir}/semantic_pred.pth')
            torch.save(occupancy_output.detach().cpu(), f'{save_dir}/occupancy_pred.pth')

            render_depth = render_results['render_depth']
            rgb_pred = render_results['render_rgb'] * 255.0
            semantic_pred = render_results['render_semantic']

            render_gt_semantic = kwargs.get('render_gt_semantic', None)
            render_gt_depth = kwargs.get('render_gt_depth', None)

            semantic = semantic_pred.argmax(2)
            semantic = OCC3D_PALETTE[semantic].to(semantic_pred)
            # visualize_image_pairs(
            #     img_inputs[0],
            #     semantic[0], # rgb_pred[0].permute(0, 2, 3, 1),
            #     render_depth[0],
            #     semantic_is_sparse=False,
            #     depth_is_sparse=False,
            #     save_dir=save_dir
            # )

            target_size = (semantic.shape[2], semantic.shape[3])  # (H, W)
            visualize_elements(
                [
                    VisElement(
                        img_inputs[0],
                        type='rgb'
                    ),
                    VisElement(
                        render_depth[0],
                        type='depth',
                        is_sparse=False,
                    ),
                    VisElement(
                        semantic[0],
                        type='semantic',
                        is_sparse=False,
                    ),
                ],
                target_size=target_size,
                save_dir=save_dir
            )
            exit()
        return render_results
    
    def loss(self, preds_dict, targets):
        if self.use_depth_consistency:
            ## 1) Compute the reprojected rgb images based on the rendered depth
            self.generate_image_pred(targets, preds_dict)

            ## 2) Compute the depth consistency loss
            loss_dict = self.compute_self_supervised_losses(targets, preds_dict)
        else:
            loss_dict = self.render_head.loss(preds_dict, targets)
        return loss_dict

    def compute_self_supervised_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        loss = 0

        depth = outputs["render_depth"]  # (M, 1, h, w)
        disp = 1.0 / (depth + 1e-7)
        color = outputs["target_imgs"]
        target = outputs["target_imgs"]

        reprojection_losses = []
        for frame_id in range(len(outputs['color_reprojection'])):
            pred = outputs['color_reprojection'][frame_id]
            reprojection_losses.append(self.compute_reprojection_loss(pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)  # (M, 2, h, w)

        ## automasking
        identity_reprojection_losses = []
        for frame_id in range(len(outputs['color_reprojection'])):
            pred = inputs["color_source_imgs"][frame_id]
            identity_reprojection_losses.append(
                self.compute_reprojection_loss(pred, target))

        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

        if self.opt.avg_reprojection:
            identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
        else:
            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses

        if self.opt.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not self.opt.disable_automasking:
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(norm_disp, color)

        loss += self.opt.disparity_smoothness * smooth_loss
        
        total_loss += loss
        losses["loss_depth_ct"] = self.depth_loss_weight * total_loss  # depth consistency loss
        return losses
    
    def generate_image_pred(self, inputs, outputs):
        color_source_imgs_list = []
        for idx in range(inputs['source_imgs'].shape[1]):
            color_source = inputs['source_imgs'][:, idx]
            color_source = rearrange(color_source, 'b num_view c h w -> (b num_view) c h w')
            color_source_imgs_list.append(color_source)
        inputs['color_source_imgs'] = color_source_imgs_list

        inv_K = inputs['inv_K'][:, self.render_view_indices]
        K = inputs['K'][:, self.render_view_indices]
        inv_K = rearrange(inv_K, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')
        K = rearrange(K, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')

        # rescale the rendered depth
        depth = outputs['render_depth'][:, self.render_view_indices]
        depth = rearrange(depth, 'b num_view h w -> (b num_view) () h w')
        depth = F.interpolate(
            depth, self.depth_ssl_size, mode="bilinear", align_corners=False)
        outputs['render_depth'] = depth

        cam_T_cam = inputs["cam_T_cam"][:, :, self.render_view_indices]

        ## 1) Depth to camera points
        cam_points = self.backproject_depth(depth, inv_K)  # (M, 4, h*w)
        len_temporal = cam_T_cam.shape[1]
        color_reprojection_list = []
        for frame_id in range(len_temporal):
            T = cam_T_cam[:, frame_id]
            T = rearrange(T, 'b num_view dim4 Dim4 -> (b num_view) dim4 Dim4')
            ## 2) Camera points to adjacent image points
            pix_coords = self.project_3d(cam_points, K, T)  # (M, h, w, 2)

            ## 3) Reproject the adjacent image
            color_source = inputs['color_source_imgs'][frame_id]  # (M, 3, h, w)
            color_reprojection = F.grid_sample(
                color_source,
                pix_coords,
                padding_mode="border", align_corners=True)
            color_reprojection_list.append(color_reprojection)

        outputs['color_reprojection'] = color_reprojection_list
        outputs['target_imgs'] = rearrange(
            inputs['target_imgs'], 'b num_view c h w -> (b num_view) c h w')

    def compute_reprojection_loss(self, pred, target, no_ssim=False):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    
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
        