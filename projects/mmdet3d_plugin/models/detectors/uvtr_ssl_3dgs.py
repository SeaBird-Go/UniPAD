'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-07-11 10:42:14
Email: haimingzhang@link.cuhk.edu.cn
Description: Use the 3DGS as the pretraining decoder.
'''
import os
import os.path as osp
import numpy as np
import pickle
import torch
from re import I
from collections import OrderedDict
import torch
import torch.nn as nn

from mmcv.cnn import Conv2d
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from pdb import set_trace
import pickle
import numpy as np
from ..utils.uni3d_voxelpooldepth import DepthNet

from .uvtr_ssl import UVTRSSL


@DETECTORS.register_module()
class UVTRSSL3DGS(UVTRSSL):
    """UVTRSSL3DGS."""

    def __init__(
        self,
        **kwargs,
    ):
        super(UVTRSSL3DGS, self).__init__(**kwargs)

    def forward_train(self, 
                      points=None, 
                      img_metas=None, 
                      img=None,
                      **kwargs):
        """Forward training function.
        Returns:
            dict: Losses of different branches.
        """
        pts_feats, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=img_metas
        )
        losses = dict()
        losses_pts = self.forward_pts_train(
            pts_feats, img_feats, points, img, img_metas, img_depth,
            **kwargs
        )
        losses.update(losses_pts)
        return losses
    
    @force_fp32(apply_to=("pts_feats", "img_feats"))
    def forward_pts_train(
        self, pts_feats, img_feats, points, 
        img, img_metas, img_depth, **kwargs
    ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        out_dict = self.pts_bbox_head(
            pts_feats, img_feats, 
            img_metas, img_depth, 
            img, **kwargs
        )
        losses = self.pts_bbox_head.loss(out_dict, dict(**kwargs))
        if self.with_depth_head and hasattr(self.pts_bbox_head.view_trans, "loss"):
            losses.update(
                self.pts_bbox_head.view_trans.loss(img_depth, points, img, img_metas)
            )
        return losses

    def simple_test(self, img_metas, points=None, img=None, **kwargs):
        """Test function without augmentaiton."""
        pts_feat, img_feats, img_depth = self.extract_feat(
            points=points, img=img, img_metas=img_metas
        )
        self.pts_bbox_head.vis_pred = True
        results = self.pts_bbox_head(
            pts_feat, img_feats, img_metas, 
            img_depth, img, **kwargs
        )
        set_trace()
        return results