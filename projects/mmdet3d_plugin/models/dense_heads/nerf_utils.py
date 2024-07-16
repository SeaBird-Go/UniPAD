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
            semantic: num_camera, H, W, 3
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
            ax[ij[i][0], ij[i][1]].imshow(concated_image_list[i][..., ::-1])
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