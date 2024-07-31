'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-07-09 16:57:18
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
import os
import os.path as osp
import numpy as np
import pickle
import torch
from nuscenes import NuScenes
from tqdm import tqdm
import mmengine
import mmcv


def update_infos(split=['train', 'val'],
                 add_occ_path=True,
                 add_scene_token=True,):
    """Add the absolute occupancy path in the Occ3D dataset to the pickle file.

    Args:
        split (list, optional): _description_. Defaults to ['train', 'val'].
    """
    if not isinstance(split, list):
        split = [split]
    
    nuscenes_version = 'v1.0-trainval'
    dataroot = './data/nuscenes/'
    nuscenes = NuScenes(nuscenes_version, dataroot)
    
    for s in split:
        assert s in ['train', 'val']
        pickle_file = f'data/nuscenes/nuscenes_unified_infos_{s}.pkl'
        print(f'Start process {pickle_file}...')

        data = mmengine.load(pickle_file)
        data_infos = data['infos']

        for info in mmcv.track_iter_progress(data_infos):
            sample = nuscenes.get('sample', info['token'])
            scene = nuscenes.get('scene', sample['scene_token'])
            
            if add_occ_path:
                info['occ_path'] = \
                    './data/nuscenes/gts/%s/%s'%(scene['name'], info['token'])
            if add_scene_token:
                info['scene_token'] = sample['scene_token']
        
        ## start saving the pickle file
        filename, ext = osp.splitext(osp.basename(pickle_file))
        root_path = "./data/nuscenes"
        new_filename = f"{filename}_v2{ext}"
        info_path = osp.join(root_path, new_filename)
        print(f"The results will be saved into {info_path}")
        mmcv.dump(data, info_path)


def add_lidarseg_path(split=['train', 'val']):
    if not isinstance(split, list):
        split = [split]

    for s in split:
        assert s in ['train', 'val']
        pickle_file = f'data/nuscenes/nuscenes_unified_infos_{s}_occ.pkl'
        print(f'Start process {pickle_file}...')

        data = mmengine.load(pickle_file)
        data_infos = data['infos']

        bevdet_pickle_file = f'data/nuscenes/bevdetv3-lidarseg-nuscenes_infos_{s}.pkl'
        bevdet_data = mmengine.load(bevdet_pickle_file)
        bevdet_data_infos = bevdet_data['infos']

        assert len(data_infos) == len(bevdet_data_infos)

        for idx in mmcv.track_iter_progress(range(len(data_infos))):
            assert data_infos[idx]['token'] == bevdet_data_infos[idx]['token']
            data_infos[idx]['lidarseg'] = bevdet_data_infos[idx]['lidarseg']
        
        ## start saving the pickle file
        filename, ext = osp.splitext(osp.basename(pickle_file))
        root_path = "./data/nuscenes"
        new_filename = f"{filename}_v2{ext}"
        info_path = osp.join(root_path, new_filename)
        print(f"The results will be saved into {info_path}")
        mmcv.dump(data, info_path)


if __name__ == '__main__':
    pickle_file = 'data/nuscenes/nuscenes_unified_infos_train.pkl'
    pickle_file = 'data/nuscenes/nuscenes_unified_infos_val.pkl'
    update_infos(['train', 'val'], add_occ_path=True, add_scene_token=True)
    # add_lidarseg_path(['train', 'val'])
    exit()
    data = mmengine.load(pickle_file)
    data_infos = data['infos']
    print(data['metadata'])
    print(len(data_infos))
