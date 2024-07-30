import mmcv
import numpy as np

from mmdet.datasets.builder import PIPELINES
import cv2
import torch
from mmcv.image.photometric import imnormalize
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor


@PIPELINES.register_module()
class LoadMultiViewMultiSweepImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        to_float32=False,
        sweep_num=1,
        random_sweep=False,
        color_type="unchanged",
        file_client_args=dict(backend="disk"),
    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.sweep_num = sweep_num
        self.random_sweep = random_sweep
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_image(self, img_filename):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            img_bytes = self.file_client.get(img_filename)
            image = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except ConnectionError:
            image = mmcv.imread(img_filename, self.color_type)
        if self.to_float32:
            image = image.astype(np.float32)
        return image

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        results["filename"] = filename

        imgs = [self._load_image(name) for name in filename]

        sweeps_paths = results["cam_sweeps_paths"]
        sweeps_ids = results["cam_sweeps_id"]
        sweeps_time = results["cam_sweeps_time"]
        if self.random_sweep:
            random_num = np.random.randint(0, self.sweep_num)
            sweeps_paths = [_sweep[:random_num] for _sweep in sweeps_paths]
            sweeps_ids = [_sweep[:random_num] for _sweep in sweeps_ids]
        else:
            random_num = self.sweep_num

        sweeps_imgs = []
        for cam_idx in range(len(sweeps_paths)):
            sweeps_imgs.extend(
                [imgs[cam_idx]]
                + [self._load_image(name) for name in sweeps_paths[cam_idx]]
            )

        results["sweeps_paths"] = [
            [filename[_idx]] + sweeps_paths[_idx] for _idx in range(len(filename))
        ]
        results["sweeps_ids"] = np.stack([[0] + _id for _id in sweeps_ids], axis=-1)
        results["sweeps_time"] = np.stack(
            [[0] + _time for _time in sweeps_time], axis=-1
        )
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img_ori"] = imgs
        results["img"] = sweeps_imgs
        results["img_shape"] = [img.shape for img in sweeps_imgs]
        results["ori_shape"] = [img.shape for img in sweeps_imgs]
        # Set initial values for default meta_keys
        results["pad_shape"] = [img.shape for img in sweeps_imgs]
        results["pad_before_shape"] = [img.shape for img in sweeps_imgs]
        results["scale_factor"] = 1.0
        num_channels = 1 if len(imgs[0].shape) < 3 else imgs[0].shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )

        # add sweep matrix to raw matrix
        results["lidar2img"] = [
            np.stack(
                [
                    results["lidar2img"][_idx],
                    *results["lidar2img_sweeps"][_idx][:random_num],
                ],
                axis=0,
            )
            for _idx in range(len(results["lidar2img"]))
        ]
        results["lidar2cam"] = [
            np.stack(
                [
                    results["lidar2cam"][_idx],
                    *results["lidar2cam_sweeps"][_idx][:random_num],
                ],
                axis=0,
            )
            for _idx in range(len(results["lidar2cam"]))
        ]
        results["cam_intrinsic"] = [
            np.stack(
                [
                    results["cam_intrinsic"][_idx],
                    *results["cam_sweeps_intrinsics"][_idx][:random_num],
                ],
                axis=0,
            )
            for _idx in range(len(results["cam_intrinsic"]))
        ]
        results.pop("lidar2img_sweeps")
        results.pop("lidar2cam_sweeps")
        results.pop("cam_sweeps_intrinsics")

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class PrepapreImageInputs(LoadMultiViewMultiSweepImageFromFiles):
    """Prepare the original mage inputs for the SSL.
    """
    def __init__(self, 
                 input_size, 
                 norm_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.input_size = input_size  # (h, w)
        if norm_cfg is not None:
            self.mean = norm_cfg['mean']
            self.std = norm_cfg['std']
        else:
            self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.std = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    def transform_core(self, 
                       img, 
                       img_size, # (h, w)
                       to_rgb=True):
        img = cv2.resize(img, img_size[::-1])
        img = imnormalize(np.array(img), self.mean, self.std, to_rgb)
        return img

    def __call__(self, results):
        assert 'adjacent' in results, \
            'adjacent should be in results'
        
        source_imgs_list = []
        for adj_info in results['adjacent']:
            cam_infos = adj_info["cams"]
            img_filename = [cam_info['data_path'] for cam_info in cam_infos.values()]
            img_adj = [self._load_image(name) for name in img_filename]

            # resize and normalize
            imgs = [
                self.transform_core(img, self.input_size)
                for img in img_adj
            ]
            imgs = [img.transpose(2, 0, 1) for img in imgs]
            source_imgs_list.append(np.ascontiguousarray(np.stack(imgs, axis=0)))

        results['source_imgs'] = DC(to_tensor(np.stack(source_imgs_list, axis=0)), stack=True)
        
        img_ori = results['img_ori']
        ## resize the image
        imgs = [
            self.transform_core(img, self.input_size)
            for img in img_ori
        ]

        # process multiple imgs in single frame
        imgs = [img.transpose(2, 0, 1) for img in imgs]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        results['target_imgs'] = DC(to_tensor(imgs), stack=True)

        ## process the intrinsic matrix
        assert 'K' in results, 'K should be in results'

        ori_shape = results['ori_shape'][0]
        origin_h, origin_w = ori_shape[0], ori_shape[1]
        h, w = self.input_size[0], self.input_size[1]
        results['K'][:, :, 0] *= w / origin_w
        results['K'][:, :, 1] *= h / origin_h
        results['inv_K'] = torch.pinverse(results['K'])

        return results