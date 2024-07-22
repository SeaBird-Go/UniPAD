'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-07-12 10:52:07
Email: haimingzhang@link.cuhk.edu.cn
Description: Some utility functions for NeRF.
'''
import os
import os.path as osp
import numpy as np
import cv2
import torch
import time
import datetime
from typing import Literal
import torchvision
from torchvision import transforms as T

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


class VisElement(object):
    def __init__(self, 
                 data=None, 
                 type: Literal['rgb', 'semantic', 'depth']='rgb',
                 is_sparse=False):
        assert type in ['rgb', 'semantic', 'depth'], \
            "The type should be in ['rgb', 'semantic', 'depth']"
        
        self.data = data
        self.type = type
        self.is_sparse = is_sparse


def visualize_depth(depth, 
                    mask=None, 
                    depth_min=None, 
                    depth_max=None, 
                    direct=False):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    if not direct:
        depth = 1.0 / (depth + 1e-6)
    invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
    if mask is not None:
        invalid_mask += np.logical_not(mask)
    if depth_min is None:
        depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
    if depth_max is None:
        depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
    depth[depth < depth_min] = depth_min
    depth[depth > depth_max] = depth_max
    depth[invalid_mask] = depth_max

    depth_scaled = (depth - depth_min) / (depth_max - depth_min)
    depth_scaled_uint8 = np.uint8(depth_scaled * 255)
    depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
    depth_color[invalid_mask, :] = 0

    return depth_color


def visualize_image_pairs(images, 
                          semantic, 
                          depth, 
                          semantic_is_sparse=False,
                          depth_is_sparse=False,
                          min_depth=0.5,
                          max_depth=55.0,
                          cam_order=[2,0,1,4,3,5],
                          save_dir=None,
                          enable_save_sperate=False):
    """Visualize the image, semantic map and depth map.

    Args:
        images (_type_): _description_
        semantic (_type_): (num_camera, H, W) if semantic_is_sparse is True, otherwise (num_camera, H, W, 3)
        depth (_type_): _description_
        semantic_is_sparse (bool, optional): _description_. Defaults to False.
        depth_is_sparse (bool, optional): _description_. Defaults to False.
        min_depth (float, optional): _description_. Defaults to 0.5.
        max_depth (float, optional): _description_. Defaults to 55.0.
        cam_order (list, optional): _description_. Defaults to [2,0,1,4,3,5].
        save_dir (_type_, optional): _description_. Defaults to None.
        enable_save_sperate (bool, optional): _description_. Defaults to False.
    """
    from .turbo_camp import turbo_colormap_data, normalize_depth, interpolate_or_clip

    ## 1) Convert all tensors into numpy array
    if not torch.is_tensor(images):
        images = torch.from_numpy(images)
        
    if torch.is_tensor(depth):
        depth = depth.detach().cpu().numpy()

    images = images.detach().cpu()
    
    target_size = (semantic.shape[1], semantic.shape[2])  # (H, W)

    # reorder the camera order
    images = images[cam_order]
    depth = depth[cam_order]

    ## 2) Denormalize the image
    assert images.shape[1] == 3, "The image should be in RGB format"
    img_mean = torch.Tensor([123.675, 116.28, 103.53])[None, :, None, None]
    img_std = torch.Tensor([58.395, 57.12, 57.375])[None, :, None, None]
    visual_imgs = images * img_std + img_mean

    visual_imgs = T.functional.resize(visual_imgs, target_size)

    all_vis_imgs = [visual_imgs]

    # process depth
    if depth is not None:
        if not depth_is_sparse:
            concated_depth_maps = []
            for b in range(len(depth)):
                pred_depth_color = visualize_depth(depth[b], direct=True)
                concated_depth_maps.append(
                        cv2.resize(pred_depth_color, (target_size[1], target_size[0])))
            depth = np.stack(concated_depth_maps, axis=0)[..., ::-1].copy()
            depth_img = torch.from_numpy(depth).permute(0, 3, 1, 2)  # to (b, 3, h, w)
        else:
            normalized_depth = normalize_depth(
                depth, d_min=min_depth, d_max=max_depth)  # (N, H, W)
            
            concated_depth_maps = []
            for _depth in normalized_depth:
                depth_rgb = np.ones((_depth.shape[0], _depth.shape[1], 3)) * 255

                valid_depth = _depth > 0.0  # the valid depth mask
                coord = np.where(valid_depth)
                for idx, depth_val in enumerate(_depth[valid_depth]):
                    depth_color = interpolate_or_clip(colormap=turbo_colormap_data, x=depth_val)
                    depth_color = (np.array(depth_color) * 255).astype(np.uint8)
                    cv2.circle(depth_rgb, 
                               (coord[1][idx], coord[0][idx]), 
                               1, depth_color.tolist(), -1)

                concated_depth_maps.append(depth_rgb)
            
            depth_img = np.stack(concated_depth_maps, axis=0)
            depth_img = torch.from_numpy(depth_img).permute(0, 3, 1, 2)

        all_vis_imgs.append(depth_img)
    
    ## Process the semantic map
    if semantic is not None:
        if not torch.is_tensor(semantic):
            semantic = torch.from_numpy(semantic)
        else:
            semantic = semantic.detach().cpu()
        
        semantic = semantic[cam_order]

        if semantic_is_sparse:
            # visualize sparse semantic map, for example, projected from the LiDAR segmentation
            concated_semantic_maps = []
            for _sem in semantic:  # loop over each camera
                semantic_rgb = np.ones((_sem.shape[0], _sem.shape[1], 3)) * 255

                valid_sem = _sem != 255
                sem_color_val = OCC3D_PALETTE[_sem[valid_sem].reshape(-1)].numpy().astype(np.uint8)

                coord = np.where(valid_sem)
                for idx, color in enumerate(sem_color_val):
                    cv2.circle(semantic_rgb, 
                               (coord[1][idx], coord[0][idx]), 
                               1, color.tolist(), -1)

                concated_semantic_maps.append(semantic_rgb)
            semantic_img = np.stack(concated_semantic_maps, axis=0)
            semantic_img = torch.from_numpy(semantic_img).permute(0, 3, 1, 2)
        else:
            semantic = semantic.permute(0, 3, 1, 2)
            semantic_img = T.functional.resize(semantic, target_size)
        
        all_vis_imgs.append(semantic_img)

    ## Start saving
    visual_imgs = torch.cat(all_vis_imgs, dim=0)

    os.makedirs(save_dir, exist_ok=True)
    file_name = str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    full_img_path = osp.join(save_dir, f'{file_name}.png')
    # need transform the image into [0, 1] with RGB format
    torchvision.utils.save_image(visual_imgs / 255.0, full_img_path, 
                                 nrow=3, padding=3, pad_value=1.0)

def visualize_image_semantic_depth_pair(images, 
                                        semantic, 
                                        depth, 
                                        cam_order=[2,0,1,4,3,5],
                                        save_dir=None,
                                        enable_save_sperate=False):
        '''
        Visualize the camera image, semantic map and dense depth map.
        Args:
            images: num_camera, 3, H, W
            semantic: num_camera, H, W, 3,  semantic map already in RGB format
            depth: num_camera, H, W
        '''
        import matplotlib.pyplot as plt

        concated_render_list = []
        concated_image_list = []

        # reorder the camera order
        images = images[cam_order]
        semantic = semantic[cam_order]
        depth = depth[cam_order]
        
        ## check if is Tensor, if not, convert to Tensor
        if torch.is_tensor(semantic):
            semantic = semantic.detach().cpu().numpy()
        
        if torch.is_tensor(depth):
            depth = depth.detach().cpu().numpy()

        for b in range(len(images)):
            visual_img = cv2.resize(images[b].transpose((1, 2, 0)), 
                                    (semantic.shape[-2], semantic.shape[-3]))
            img_mean = np.array([0.485, 0.456, 0.406])[None, None, :]
            img_std = np.array([0.229, 0.224, 0.225])[None, None, :]
            visual_img = np.ascontiguousarray((visual_img * img_std + img_mean))
            concated_image_list.append(visual_img)  # convert to [0, 255] scale
            
            # visualize the depth
            pred_depth_color = visualize_depth(depth[b], direct=True)
            pred_depth_color = pred_depth_color[..., [2, 1, 0]]
            concated_render_list.append(
                cv2.resize(pred_depth_color.copy(), 
                           (semantic.shape[-2], semantic.shape[-3])))

        fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(6, 6))
        ij = [[i, j] for i in range(2) for j in range(3)]
        for i in range(len(ij)):
            ax[ij[i][0], ij[i][1]].imshow(concated_image_list[i])
            ax[ij[i][0] + 2, ij[i][1]].imshow(semantic[i] / 255)
            ax[ij[i][0] + 4, ij[i][1]].imshow(concated_render_list[i] / 255)

            for j in range(3):
                ax[i, j].axis('off')

        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        ## save the seperate images
        if save_dir is not None:
            from PIL import Image
            os.makedirs(save_dir, exist_ok=True)

            full_img_path = osp.join(save_dir, '%f.png' % time.time())
            plt.savefig(full_img_path)

            if enable_save_sperate:
                for i in range(len(concated_render_list)):
                    depth_map = concated_render_list[i].astype(np.uint8)
                    semantic_map = semantic[i].astype(np.uint8)
                    camera_img = (concated_image_list[i][..., ::-1] * 255.0).astype(np.uint8)

                    save_depth_map_path = osp.join(save_dir, f"{i:02}_rendered_depth.png")
                    save_semantic_map_path = osp.join(save_dir, f"{i:02}_rendered_semantic.png")
                    save_camera_img_path = osp.join(save_dir, f"{i:02}_camera_img.png")

                    depth_map = Image.fromarray(depth_map)
                    depth_map.save(save_depth_map_path)

                    semantic_map = Image.fromarray(semantic_map)
                    semantic_map.save(save_semantic_map_path)

                    camera_img = Image.fromarray(camera_img)
                    camera_img.save(save_camera_img_path)

                    ## save the depth map
                    rendered_depth = depth[i]
                    save_depth_path = osp.join(save_dir, f"{i:02}_rendered_depth.npy")
                    np.save(save_depth_path, rendered_depth)
        else:
            plt.show()