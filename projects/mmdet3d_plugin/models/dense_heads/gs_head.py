'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-07-10 10:45:56
Email: haimingzhang@link.cuhk.edu.cn
Description: The 3D Gaussian Splatting rendering head.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import os.path as osp
from einops import rearrange, repeat
from mmdet3d.models.builder import HEADS
from mmcv.runner.base_module import BaseModule
# from mmdet3d.models.decode_heads.nerf_head import NeRFDecoderHead
from .common.gaussians import build_covariance
from .common.cuda_splatting import render_cuda, render_depth_cuda, render_depth_cuda2
from .common.sh_rotation import rotate_sh


class Gaussians:
    means: torch.FloatTensor
    covariances: torch.FloatTensor
    scales: torch.FloatTensor
    rotations: torch.FloatTensor
    harmonics: torch.FloatTensor
    opacities: torch.FloatTensor


@HEADS.register_module()
class GaussianSplattingDecoder(BaseModule):
    def __init__(self,
                 semantic_head=False,
                 render_size=None,
                 depth_range=None,
                 pc_range=None,
                 voxels_size=None,
                 step_size=1,
                 learn_gs_scale_rot=False,
                 gs_scale_min=0.1,
                 gs_scale_max=0.24,
                 sh_degree=4,
                 volume_size=(200, 200, 16),
                 in_channels=32,
                 num_surfaces=1,
                 offset_scale=0.05,
                 gs_scale=0.05,
                 **kwargs):
        super().__init__()

        self.render_h, self.render_w = render_size
        self.min_depth, self.max_depth = depth_range

        self.semantic_head = semantic_head

        self.learn_gs_scale_rot = learn_gs_scale_rot
        self.offset_scale = offset_scale
        self.gs_scale = gs_scale

        self.voxels_size = voxels_size
        self.stepsize = step_size
        
        self.xyz_min = torch.from_numpy(np.array(pc_range[:3]))  # (x_min, y_min, z_min)
        self.xyz_max = torch.from_numpy(np.array(pc_range[3:]))  # (x_max, y_max, z_max)

        self.num_voxels = self.voxels_size[0] * self.voxels_size[1] * self.voxels_size[2]
        ## NOTE: the voxel_size is the size of a single voxel, while the voxels_size is a list
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels).pow(1 / 3)
        ratio_x = self.xyz_max[0] / (self.xyz_max[0] - self.xyz_min[0])
        ratio_y = self.xyz_max[1] / (self.xyz_max[1] - self.xyz_min[1])
        ratio_z = self.xyz_max[2] / (self.xyz_max[2] - self.xyz_min[2])
        N_samples = int(np.linalg.norm(np.array([self.voxels_size[0] * ratio_x, 
                                                 self.voxels_size[1] * ratio_y, 
                                                 self.voxels_size[2] * ratio_z]) + 1) / self.stepsize) + 1
        self.rng = torch.arange(N_samples)[None].float()

        ## construct the volume grid
        xs = torch.arange(
            self.xyz_min[0], self.xyz_max[0],
            (self.xyz_max[0] - self.xyz_min[0]) / volume_size[0])
        ys = torch.arange(
            self.xyz_min[1], self.xyz_max[1],
            (self.xyz_max[1] - self.xyz_min[1]) / volume_size[1])
        zs = torch.arange(
            self.xyz_min[2], self.xyz_max[2],
            (self.xyz_max[2] - self.xyz_min[2]) / volume_size[2])
        W, H, D = len(xs), len(ys), len(zs)

        xyzs = torch.stack([
            xs[None, :, None].expand(H, W, D),
            ys[:, None, None].expand(H, W, D),
            zs[None, None, :].expand(H, W, D)
        ], dim=-1).permute(1, 0, 2, 3)  # (200, 200, 16, 3)

        # the volume grid coordinates in ego frame
        self.volume_xyz = xyzs.to(torch.float32)

        self.gs_scale_min = gs_scale_min
        self.gs_scale_max = gs_scale_max
        self.d_sh = (sh_degree + 1) ** 2

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        self.OCC3D_PALETTE = torch.Tensor([
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

        self.to_gaussians = nn.Sequential(
            nn.ReLU(),
            nn.Linear(
                in_channels,
                num_surfaces * (3 + 3 + 4 + 3 * self.d_sh)
            )
        )

    def forward(self, 
                inputs,
                **kwargs):
        """Foward function

        Args:
            density_prob (Tensor): (bs, 1, 200, 200, 16)
            rgb_recon (Tensor): (bs, 3, 200, 200, 16)
            occ_semantic (Tensor): (bs, c, 200, 200, 16)
            intricics (Tensor): (bs, num_view, 4, 4)
            pose_spatial (Tensor): (bs, num_view, 4, 4)
            volume_feat (Tensor): (bs, 200, 200, 16, c)
            render_mask (_type_, optional): _description_. Defaults to None.
            vis_semantic (bool, optional): Use for visualization debug. Defaults to False.

        Returns:
            Tuple: rendered depth, rgb images and semantic features
        """
        # get occupancy features
        density_prob, semantic, volume_feat = self.get_occ_features(inputs)

        intricics, pose_spatial = inputs['intrinsics'], inputs['pose_spatial']

        render_depth, render_rgb, render_semantic = self.train_gaussian_rasterization_v2(
            density_prob,
            None,
            semantic,
            intricics,
            pose_spatial,
            volume_feat=volume_feat
        )
        render_depth = render_depth.clamp(self.min_depth, self.max_depth)
        return render_depth, render_rgb, render_semantic

    def get_occ_features(self, inputs):
        occupancy = inputs['occupancy_score']  # B, 1, X, Y, Z
        B, _, X, Y, Z = occupancy.shape
        semantic = inputs['semantic'] # B, C, X, Y, Z
        feature_map = inputs['volume_feat']  # B, C, X, Y, Z
        return occupancy, semantic, feature_map
    
    def train_gaussian_rasterization(self, 
                                     density_prob, 
                                     rgb_recon, 
                                     semantic_pred, 
                                     intrinsics, 
                                     extrinsics, 
                                     render_mask=None,
                                     vis_semantic=False,
                                     **kwargs):
        b, v = intrinsics.shape[:2]
        device = density_prob.device
        
        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[..., 0, :] /= self.render_w
        intrinsics[..., 1, :] /= self.render_h

        transform = torch.Tensor([[0, 1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]).to(device)
        extrinsics = transform.unsqueeze(0).unsqueeze(0) @ extrinsics
        
        bs = density_prob.shape[0]

        xyzs = repeat(self.volume_xyz, 'h w d dim3 -> bs h w d dim3', bs=bs).to(device)
        xyzs = rearrange(xyzs, 'b h w d dim3 -> b (h w d) dim3') # (bs, num, 3)

        if self.semantic_head:
            semantic_pred = rearrange(semantic_pred, 'b c h w d -> b (h w d) c').float()
            # semantic_pred = repeat(semantic_pred, 'b xyz c -> b xyz (dim17 c)', dim17=17)

        density_prob = rearrange(density_prob, 'b dim1 h w d -> (b dim1) (h w d)')
        
        if vis_semantic:
            harmonics = rearrange(rgb_recon, 'b dim3 h w d -> b (h w d) dim3 ()')
        else:
            ## TODO: currently the harmonics is a dummy variable when training
            harmonics = self.OCC3D_PALETTE[torch.argmax(rgb_recon, dim=1).long()].to(device)
            harmonics = rearrange(harmonics, 'b h w d dim3 -> b (h w d) dim3 ()')

        g = xyzs.shape[1]

        gaussians = Gaussians
        gaussians.means = xyzs  ######## Gaussian center ########
        gaussians.opacities = torch.sigmoid(density_prob) ######## Gaussian opacities ########

        scales = torch.ones(3).unsqueeze(0).to(device) * 0.2
        rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).to(device)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances ######## Gaussian covariances ########

        gaussians.harmonics = harmonics ######## Gaussian harmonics ########

        render_results = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b v i j -> (b v) g i j", g=g),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=False,
            feats3D=repeat(semantic_pred, "b g c -> (b v) g c", v=v) if self.semantic_head else None,
        )
        if self.semantic_head:
            color, depth, feats = render_results
            feats = rearrange(feats, "(b v) c h w -> b v c h w", b=b, v=v)
        else:
            color, depth = render_results
            feats = None
        
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2)

        return depth, color, feats
    
    def loss(self,
             pred_dict,
             target_dict):
        
        losses = dict()

        render_depth = pred_dict['render_depth']
        gt_depth = target_dict['render_gt_depth']

        loss_render_depth = self.compute_depth_loss(
            render_depth, gt_depth, gt_depth > 0.0)
        if torch.isnan(loss_render_depth):
            print('NaN in render depth loss!')
            loss_render_depth = torch.Tensor([0.0]).to(render_depth.device)
        losses['loss_render_depth'] = loss_render_depth

        if self.semantic_head:
            assert 'img_semantic' in target_dict.keys()
            img_semantic = target_dict['img_semantic']

            semantic_pred = pred_dict['render_semantic']
            
            loss_render_sem = self.compute_semantic_loss(semantic_pred, img_semantic)
            if torch.isnan(loss_render_sem):
                print('NaN in render semantic loss!')
                loss_render_sem = torch.Tensor([0.0]).to(render_depth.device)
            losses['loss_render_sem'] = loss_render_sem

        return losses

    def train_gaussian_rasterization_v2(self, 
                                        density_prob, 
                                        rgb_recon, 
                                        semantic_pred, 
                                        intrinsics, 
                                        extrinsics, 
                                        volume_feat=None):
        b, v = intrinsics.shape[:2]
        device = density_prob.device
        
        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[..., 0, :] /= self.render_w
        intrinsics[..., 1, :] /= self.render_h

        # transform = torch.Tensor([[0, 1, 0, 0],
        #                           [1, 0, 0, 0],
        #                           [0, 0, 1, 0],
        #                           [0, 0, 0, 1]]).to(device)
        # extrinsics = transform.unsqueeze(0).unsqueeze(0) @ extrinsics
        
        density_prob = rearrange(density_prob, 'b dim1 h w d -> (b dim1) (h w d)')

        if self.semantic_head:
            semantic_pred = rearrange(semantic_pred, 'b c h w d -> b (h w d) c')

        gaussians = self.predict_gaussian(density_prob,
                                          extrinsics,
                                          volume_feat)
        render_results = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            rearrange(gaussians.covariances, "b v g i j -> (b v) g i j"),
            rearrange(gaussians.harmonics, "b v g c d_sh -> (b v) g c d_sh"),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=True,
            feats3D=repeat(semantic_pred, "b g c -> (b v) g c", v=v)
        )
        if self.semantic_head:
            color, depth, feats = render_results
            feats = rearrange(feats, "(b v) c h w -> b v c h w", b=b, v=v)
        else:
            color, depth = render_results
            feats = None
        
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2)

        return depth, color, feats
    

    def visualize_gaussian(self,
                           density_prob, 
                           rgb_recon, 
                           semantic_pred, 
                           intrinsics, 
                           extrinsics):
        b, v = intrinsics.shape[:2]
        device = density_prob.device
        
        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[..., 0, :] /= self.render_w
        intrinsics[..., 1, :] /= self.render_h

        transform = torch.Tensor([[0, 1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]).to(device)
        extrinsics = transform.unsqueeze(0).unsqueeze(0) @ extrinsics
        
        bs = density_prob.shape[0]
        xyzs = repeat(self.volume_xyz, 'h w d dim3 -> bs h w d dim3', bs=bs)
        xyzs = rearrange(xyzs, 'b h w d dim3 -> b (h w d) dim3') # (bs, num, 3)

        density_prob = rearrange(density_prob, 'b dim1 h w d -> (b dim1) (h w d)')

        if self.semantic_head:
            semantic_pred = rearrange(semantic_pred, 'b c h w d -> b (h w d) c')

        harmonics = rearrange(rgb_recon, 'b dim3 h w d -> b (h w d) dim3 ()')
        g = xyzs.shape[1]

        gaussians = Gaussians
        gaussians.means = xyzs  ######## Gaussian center ########
        gaussians.opacities = torch.sigmoid(density_prob) ######## Gaussian opacities ########

        scales = torch.ones(3).unsqueeze(0).to(device) * 0.2
        rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).to(device)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances ######## Gaussian covariances ########

        gaussians.harmonics = harmonics ######## Gaussian harmonics ########

        render_results = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b v i j -> (b v) g i j", g=g),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=False,
            feats3D=repeat(semantic_pred, "b g c -> (b v) g c", v=v)
        )
        if self.semantic_head:
            color, depth, feats = render_results
            feats = rearrange(feats, "(b v) c h w -> b v c h w", b=b, v=v)
        else:
            color, depth = render_results
            feats = None
        
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2)

        return depth, color, feats

    def predict_gaussian(self,
                         density_prob,
                         extrinsics,
                         volume_feat):
        
        bs, v = extrinsics.shape[:2]
        device = extrinsics.device

        xyzs = repeat(self.volume_xyz, 'h w d dim3 -> bs h w d dim3', bs=bs).to(device)
        xyzs = rearrange(xyzs, 'b h w d dim3 -> b (h w d) dim3') # (bs, num, 3)
        
        # predict the Gaussian parameters from volume feature
        raw_gaussians = self.to_gaussians(volume_feat)
        raw_gaussians = rearrange(raw_gaussians, 'b h w d c -> b (h w d) c')
        xyz_offset, scales, rotations, sh = raw_gaussians.split(
            (3, 3, 4, 3 * self.d_sh), dim=-1)
        
        # construct 3D Gaussians
        gaussians = Gaussians
        gaussians.opacities = torch.sigmoid(density_prob)
        gaussians.means = xyzs + (xyz_offset.sigmoid() - 0.5) * self.offset_scale

        # Learn scale and rotation of 3D Gaussians
        if self.learn_gs_scale_rot:
            # Set scale and rotation of 3D Gaussians
            scale_min = self.gs_scale_min
            scale_max = self.gs_scale_max
            scales = scale_min + (scale_max - scale_min) * torch.sigmoid(scales)

            # Normalize the quaternion features to yield a valid quaternion.
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            scales = torch.ones(3).unsqueeze(0).unsqueeze(0).to(device) * self.gs_scale
            rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).unsqueeze(0).to(device)
            scales = scales.repeat(bs, xyzs.shape[1], 1)
            rotations = rotations.repeat(bs, xyzs.shape[1], 1)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        covariances = rearrange(covariances, "b g i j -> b () g i j")

        c2w_rotations = extrinsics[..., :3, :3]
        c2w_rotations = rearrange(c2w_rotations, "b v i j -> b v () i j")
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances  # (bs, v, g, i, j)

        # Apply sigmoid to get valid colors.
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*gaussians.opacities.shape, 3, self.d_sh)) * self.sh_mask
        gaussians.harmonics = repeat(sh, 'b g xyz d_sh -> b v g xyz d_sh', v=v)

        return gaussians

    
    def gaussian_rasterization(self, 
                               density_prob, 
                               rgb_recon, 
                               semantic_pred, 
                               intrinsics, 
                               extrinsics, 
                               render_mask=None):
        b, v = intrinsics.shape[:2]
        
        near = torch.ones(b, v).to(density_prob.device) * 1
        far = torch.ones(b, v).to(density_prob.device) * 100
        background_color = torch.zeros((3), dtype=torch.float32).to(density_prob.device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[..., 0, :] /= self.render_w
        intrinsics[..., 1, :] /= self.render_h

        transform = torch.Tensor([[0, 1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]).to(density_prob.device)
        extrinsics = transform.unsqueeze(0).unsqueeze(0) @ extrinsics

        device = density_prob.device
        xs = torch.arange(
            self.xyz_min[0], self.xyz_max[0],
            (self.xyz_max[0] - self.xyz_min[0]) / density_prob.shape[2], device=device)
        ys = torch.arange(
            self.xyz_min[1], self.xyz_max[1],
            (self.xyz_max[1] - self.xyz_min[1]) / density_prob.shape[3], device=device)
        zs = torch.arange(
            self.xyz_min[2], self.xyz_max[2],
            (self.xyz_max[2] - self.xyz_min[2]) / density_prob.shape[4], device=device)
        W, H, D = len(xs), len(ys), len(zs)
        
        bs = density_prob.shape[0]
        xyzs = torch.stack([
            xs[None, :, None].expand(H, W, D),
            ys[:, None, None].expand(H, W, D),
            zs[None, None, :].expand(H, W, D)
        ], dim=-1)[None].expand(bs, H, W, D, 3).flatten(0, 3)
        density_prob = density_prob.flatten()

        mask = (density_prob > 0) #& (semantic_pred.flatten()==3)
        xyzs = xyzs[mask]

        harmonics = self.OCC3D_PALETTE[semantic_pred.long().flatten()].to(device)
        harmonics = harmonics[mask]

        density_prob = density_prob[mask]
        density_prob = density_prob.unsqueeze(0)
        xyzs = xyzs.unsqueeze(0)

        g = xyzs.shape[1]

        gaussians = Gaussians
        gaussians.means = xyzs  ######## Gaussian center ########
        gaussians.opacities = torch.where(density_prob>0, 1., 0.) ######## Gaussian opacities ########

        scales = torch.ones(3).unsqueeze(0).to(device) * 0.05
        rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).to(device)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances ######## Gaussian covariances ########

        harmonics = harmonics.unsqueeze(-1).unsqueeze(0)
        # harmonics = torch.ones_like(xyzs).unsqueeze(-1)
        gaussians.harmonics = harmonics ######## Gaussian harmonics ########

        color, depth = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b v i j -> (b v) g i j", g=g),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=False,
        )

        return color, depth.squeeze(1)