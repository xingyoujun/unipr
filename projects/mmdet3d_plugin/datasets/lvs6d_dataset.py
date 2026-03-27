import mmcv
import numpy as np
from mmdet.datasets import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import os

@DATASETS.register_module()
class Lvs6dDataset(Custom3DDataset):
    """
    Lvs6dDataset Dataset.
    """
    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 use_valid_flag=False):

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = data['infos']

        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.
        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            image_id=info['image_id'],
            interocular_distance=info['interocular_distance'],
            timestamp=info['timestamp'],
        )

        image_paths = []
        world2img_rts = []
        intrinsics = []
        extrinsics = []
        # mask = []
        for cam_type, cam_info in info['cams'].items():
            image_path = '/'.join(cam_info['image_path'].split('/')[3:])
            image_path = os.path.join(self.data_root, image_path)
            image_paths.append(image_path)

            intrinsic = cam_info['cam_intrinsic']
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic

            # obtain lidar to image transformation matrix
            world2cam_r = np.linalg.inv(cam_info['cam_R_w2c'])
            world2cam_t = cam_info['cam_t_w2c'] @ world2cam_r.T
            world2cam_rt = np.eye(4)
            world2cam_rt[:3, :3] = world2cam_r.T
            world2cam_rt[3, :3] = -world2cam_t

            world2img_rt = (viewpad @ world2cam_rt.T)
            intrinsics.append(viewpad)
            extrinsics.append(world2cam_rt) 
            world2img_rts.append(world2img_rt)


        input_dict.update(
            dict(
                img_filename=image_paths,
                world2img=world2img_rts,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                cam_r_w2c=info['cams']['CAM_LEFT']['cam_R_w2c'],
                cam_t_w2c=info['cams']['CAM_LEFT']['cam_t_w2c'],
                # mask=mask,
            ))

        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos
        
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        gt_labels_2d = info['gt_labels']
        gt_labels_3d = np.array([0 for _ in info['gt_labels']]).astype('long') # ignore label for 3d training
        gt_bboxes_2d = info['gt_2d_boxes']
        gt_bboxes_3d = info['gt_3d_boxes']
        
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.)).convert_to(self.box_mode_3d) # change to (0.5,0.5.0) to skip center changing
        
        if 'gt_parts' in info:
            gt_parts_3d = info['gt_parts']
            part_labels_3d = info['part_labels']
            part_bboxes = info['part_boxes']
            part_bboxes_3d = []
            for part_bbox in part_bboxes:
                part_bbox = LiDARInstance3DBoxes(part_bbox, box_dim=part_bbox.shape[-1], origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
                part_bboxes_3d.append(part_bbox)
        
        if 'gt_embeddings' in info:
            gt_embeddings = info['gt_embeddings']
            if 'gt_parts' in info:
                part_embeddings = info['part_embeddings']
            
                final_part_embeddings = []
                for part_embedding in part_embeddings:
                    final_part_embeddings.append(part_embedding.astype(np.float32))
        
        if 'vol_points_list' in info:
            gt_vol_points = info['vol_points_list'].astype(np.float32)
            gt_vol_label = info['vol_label_list'].astype(np.float32)

        anns_results = dict(
            gt_labels_2d=gt_labels_2d,
            gt_labels_3d=gt_labels_3d,
            gt_bboxes_2d=gt_bboxes_2d,
            gt_bboxes_3d=gt_bboxes_3d,)

        if 'gt_parts' in info:
            anns_results['gt_parts_3d'] = gt_parts_3d
            anns_results['part_labels_3d'] = part_labels_3d
            anns_results['part_bboxes_3d'] = part_bboxes_3d

        if 'gt_embeddings' in info:
            anns_results['gt_embeddings'] = gt_embeddings
            if 'gt_parts' in info:
                anns_results['part_embeddings'] = final_part_embeddings

        if 'vol_points_list' in info:
            anns_results['gt_vol_points'] = gt_vol_points
            anns_results['gt_vol_label'] = gt_vol_label

        return anns_results

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
