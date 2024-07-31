'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-07-26 16:22:58
Email: haimingzhang@link.cuhk.edu.cn
Description: The nuscenes dataset for self-supervised learning.
'''
import os
import os.path as osp
import numpy as np
import pickle
import torch
# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp
from pyquaternion import Quaternion

import mmdet3d
from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose

from .nuscenes_dataset import NuScenesSweepDataset


def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat


@DATASETS.register_module()
class NuScenesSweepDatasetSSL(NuScenesSweepDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """
    def __init__(
        self,
        use_depth_consistency=False,
        extra_frames=[-1, 1],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.use_depth_consistency = use_depth_consistency
        self.extra_frames = extra_frames

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        input_dict = super().get_data_info(index)

        if self.use_depth_consistency:
            curr_cam_to_ego = np.stack(input_dict['cam2camego'])  # (6, 4, 4)
            curr_camego_to_global = np.stack(input_dict['camego2global'])  # (6, 4, 4)

            cam_intrinsic = np.stack(input_dict['cam_intrinsic'])
            input_dict['K'] = torch.from_numpy(cam_intrinsic).to(torch.float32)  # (6, 4, 4)
            
            seq_cam_to_cam_list = []
            info_adj_list = []
            for idx in self.extra_frames:
                adj_idx = index + idx
                
                select_id = max(min(adj_idx, len(self) - 1), 0)
                adj_info = self.data_infos[select_id]
                curr_info = self.data_infos[index]
                if adj_info['scene_token'] != curr_info['scene_token']:
                    adj_info = curr_info
                
                info_adj_list.append(adj_info)

                adj_global_to_ego_list,  adj_ego_to_cam_list = [], []
                for cam_type, cam_info in adj_info["cams"].items():
                    cam_to_ego = rt2mat(cam_info['sensor2ego_translation'],
                                        cam_info['sensor2ego_rotation'])
                    ego_to_cam = np.linalg.inv(cam_to_ego)
                    
                    ego_to_global = rt2mat(cam_info['ego2global_translation'],
                                           cam_info['ego2global_rotation'])
                    global_to_ego = np.linalg.inv(ego_to_global)

                    adj_global_to_ego_list.append(global_to_ego)
                    adj_ego_to_cam_list.append(ego_to_cam)
                
                adj_global_to_ego = np.stack(adj_global_to_ego_list)
                adj_ego_to_cam = np.stack(adj_ego_to_cam_list)

                curr_cam_to_cami = adj_ego_to_cam @ adj_global_to_ego @ curr_camego_to_global @ curr_cam_to_ego
                seq_cam_to_cam_list.append(curr_cam_to_cami)
            
            cam_T_cam = np.stack(seq_cam_to_cam_list)
            input_dict['cam_T_cam'] = torch.from_numpy(cam_T_cam).to(torch.float32)  # (2, 6, 4, 4)
            input_dict.update(dict(adjacent=info_adj_list))
        return input_dict
