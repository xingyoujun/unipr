import mmcv
import numpy as np
import os
import json
from pycocotools.coco import COCO
from os import path as osp
from pyquaternion import Quaternion

loc_max = [0,0,0,0,-1,200]

def create_lvs6d_infos(root_path, split_name, debug_num=20):
    """
    Create info file of lvs6d dataset.

    Given the raw data, generate its related info file in pkl format.
    
    """
    json_name = "coco_annotations.json"

    left_dataset = COCO(os.path.join(root_path, f'lvs6d_{split_name}', 's3d_data_left', json_name))
    right_dataset = COCO(os.path.join(root_path, f'lvs6d_{split_name}', 's3d_data_right', json_name))
    assert len(left_dataset.imgs) == len(right_dataset.imgs)
    print('total imgs num: {}'.format(len(left_dataset.imgs)))

    with np.load(os.path.join(root_path, f'latents_{split_name}.npz'), allow_pickle=True) as data:
        latents = data['latents']
        latents = latents.item()

    with np.load(os.path.join(root_path, 'lvs6d_pc_center.npz'), allow_pickle=True) as data:
        pc_centers = data['pc_centers']
        pc_centers = pc_centers.item()

    with np.load(os.path.join(root_path, 'lvs6d_scale.npz'), allow_pickle=True) as data:
        scales = data['scales']
        scales = scales.item()

    with open(os.path.join(root_path, 'category_info.json'), 'r', encoding='utf-8') as f:
        category_data = json.load(f)

    name_to_idx = {info["category_name"]: int(idx) for idx, info in category_data.items()}

    lvs6d_infos = []

    for image_id in mmcv.track_iter_progress(range(debug_num)):
        l_image = left_dataset.imgs[image_id]
        l_annos = left_dataset.getAnnIds(image_id)
        l_annos = left_dataset.loadAnns(l_annos)
        l_img_path = os.path.join(root_path, f'lvs6d_{split_name}', 's3d_data_left', l_image['file_name'])

        l_cam_K = np.array(l_image['cam_K']).reshape(3,3)
        l_cam_R_w2c = np.array(l_image['cam_R_w2c']).reshape(3,3)
        l_cam_t_w2c = np.array(l_image['cam_t_w2c'])

        r_image = right_dataset.imgs[image_id]
        r_annos = right_dataset.getAnnIds(image_id)
        r_annos = right_dataset.loadAnns(r_annos)
        r_img_path = os.path.join(root_path, f'lvs6d_{split_name}', 's3d_data_right', r_image['file_name'])

        r_cam_K = np.array(r_image['cam_K']).reshape(3,3)
        r_cam_R_w2c = np.array(r_image['cam_R_w2c']).reshape(3,3)
        r_cam_t_w2c = np.array(r_image['cam_t_w2c'])

        if 'interocular_distance' in l_image:
            interocular_distance = l_image['interocular_distance']
        else:
            interocular_distance = 13

        r_cam_t_w2c[1] -= interocular_distance
        
        mmcv.check_file_exist(l_img_path)
        mmcv.check_file_exist(r_img_path)

        info = {
            'image_id': image_id,
            'cams': dict(),
            'interocular_distance': interocular_distance,
            'timestamp': l_image['date_captured'],
        }

        l_obj = np.array([b['object_name'] for b in l_annos])
        r_obj = np.array([b['object_name'] for b in r_annos])

        tmp = []
        for idx, l_anno in enumerate(l_annos):
            if l_anno['object_name'] in r_obj:
                tmp.append(l_anno)
        l_annos = tmp
        tmp = []
        for idx, r_anno in enumerate(r_annos):
            if r_anno['object_name'] in l_obj:
                tmp.append(r_anno)
        r_annos = tmp

        l_obj = np.array([b['object_name'] for b in l_annos])
        r_obj = np.array([b['object_name'] for b in r_annos])
        
        labels = np.array([b['category_id'] for b in l_annos])
        bboxes_2d_left = np.array([b['bbox'] for b in l_annos]).reshape(-1, 4)
        bboxes_2d_left[:,2:] += bboxes_2d_left[:,:2]
        bboxes_2d_right = np.array([b['bbox'] for b in r_annos]).reshape(-1, 4)
        bboxes_2d_right[:,2:] += bboxes_2d_right[:,:2]
        bboxes_2d = np.concatenate((bboxes_2d_left,bboxes_2d_right),axis=-1)
        locs = np.array([b['center'] for b in l_annos]).reshape(-1, 3)
        
        anno_ids = np.array([b['id'] for b in l_annos])

        if len(locs) != 0:
            loc_max[0] = max(max(locs[...,0]), loc_max[0])
            loc_max[1] = min(min(locs[...,0]), loc_max[1])
            loc_max[2] = max(max(locs[...,1]), loc_max[2])
            loc_max[3] = min(min(locs[...,1]), loc_max[3])
            loc_max[4] = max(max(locs[...,2]), loc_max[4])
            loc_max[5] = min(min(locs[...,2]), loc_max[5])

        dims = np.array([b['dimensions'] for b in l_annos]).reshape(-1, 3)
        rots = np.array([Quaternion(b['orientation']).rotation_matrix for b in l_annos]).reshape(-1, 3, 3)
        
        mask_flags = []
        # l_mask = np.zeros((600,960), dtype=int)
        # r_mask = np.zeros((600,960), dtype=int)
        for idx in range(len(l_annos)):
            mask_flag = True

            if l_obj[idx] == 'bread':
                mask_flag = False

            # rle = l_annos[idx]['segmentation']['counts']
            # assert sum(rle) == l_annos[idx]['segmentation']['size'][0] * l_annos[idx]['segmentation']['size'][1]
            # tmp = left_dataset.annToMask(l_annos[idx]) == 1
            if l_annos[idx]['area'] < 200:
                mask_flag = False
            
            # rle = r_annos[idx]['segmentation']['counts']
            # assert sum(rle) == r_annos[idx]['segmentation']['size'][0] * r_annos[idx]['segmentation']['size'][1]
            # tmp = right_dataset.annToMask(r_annos[idx]) == 1
            if r_annos[idx]['area'] < 200:
                mask_flag = False

            mask_flags.append(mask_flag)

        if len(mask_flags) == 0:
            mask_flags = np.array([]).astype(np.long)
        else:
            mask_flags = np.stack(mask_flags, axis=0)
        
        gt_labels = []
        gt_2d_boxes = []
        gt_3d_boxes = []
        gt_embeddings = []
        vol_points_list = []
        vol_label_list = []
        l_obj = l_obj[mask_flags]
        labels = labels[mask_flags]
        bboxes_2d = bboxes_2d[mask_flags]
        locs = locs[mask_flags]
        dims = dims[mask_flags]
        rots = rots[mask_flags]
        anno_ids = anno_ids[mask_flags]

        assert len(l_obj) == len(bboxes_2d) == len(labels) == len(locs)
        for idx, name in enumerate(l_obj):

            custom_obj_list = ['cereal_bottle', 'clip_2000', "fork_2000", "frying_spoon_2000","jam_bottle","knife_2000","knife_holder","large_bowl_2000",
                        "large_plate_2000","large_spoon_2000", "lunch_plate_2000","meal_spoon_2000","middle_plate_2000","milk_bottle","mug_cup_2000",
                        "slotted_spoon_2000", "small_bowl_2000","small_spoon_2000","soup_bowl_2000","soup_spoon_2000","spatula_2000","spoon","square_plate_2000","tray_2000"]

            if any(name == name_ for name_ in custom_obj_list):
                name_dir = 'tx'
            else:
                import re
                name_dir = re.findall(r'^([a-zA-Z_]+)(?=\d)', name)
                name_dir = name_dir[0][:-1]
                if 'lbox_part' in name_dir:
                    name_dir = name_dir.replace('lbox_part', 'lbox')
            
            gt_labels.append(name_to_idx[name_dir])
            gt_2d_boxes.append(bboxes_2d[idx])
            
            scale = scales[name] * 100
            pc_center = pc_centers[name]
            
            rot = rots[idx].astype(np.float32)

            voxel_path = os.path.join('/home/zcr/xyj/Project/3DShape2VecSet/data/coders/gt/', 'voxel_'+ name + '.npz')
            try:
                with np.load(voxel_path) as data:
                    vol_points = data['vol_points'].astype(np.float32)
                    vol_label = data['vol_label'].astype(np.float32)
            except Exception as e:
                print(e)
                print(voxel_path)

            ind = np.random.default_rng().choice(vol_points.shape[0], 2048, replace=False)
            vol_points = vol_points[ind]
            vol_label = vol_label[ind]

            vol_points = np.einsum('ij,kj->ki', rot, vol_points)

            vol_points_list.append(vol_points)
            vol_label_list.append(vol_label)

            center = locs[idx] + pc_center * 100
            scale_2d = max((gt_2d_boxes[0][3] - gt_2d_boxes[0][1]), (gt_2d_boxes[0][2] - gt_2d_boxes[0][0]))
            gt_box = np.concatenate([center, [scale], [scale_2d]], axis=0)

            mean, logvar = latents[anno_ids[idx]]
            embedding = np.concatenate([mean,logvar])

            gt_3d_boxes.append(gt_box)
            gt_embeddings.append(embedding)

        gt_labels = np.array(gt_labels)
        gt_2d_boxes = np.array(gt_2d_boxes)
        gt_3d_boxes = np.array(gt_3d_boxes)
        gt_embeddings = np.array(gt_embeddings)

        vol_points_list = np.array(vol_points_list)
        vol_label_list = np.array(vol_label_list)

        info['cams']['CAM_LEFT'] = {
            'image_path': l_img_path,
            'cam_intrinsic': l_cam_K,
            'cam_R_w2c': l_cam_R_w2c,
            'cam_t_w2c': l_cam_t_w2c,
            #'mask': l_mask,
        }

        info['cams']['CAM_RIGHT'] = {
            'image_path': r_img_path,
            'cam_intrinsic': r_cam_K,
            'cam_R_w2c': r_cam_R_w2c,
            'cam_t_w2c': r_cam_t_w2c,
            #'mask': r_mask,
        }
        
        if len(gt_3d_boxes) != 0:
            info['gt_labels'] = gt_labels
            info['gt_2d_boxes'] = gt_2d_boxes.astype(np.float32)
            info['gt_3d_boxes'] = gt_3d_boxes.astype(np.float32)
            info['gt_embeddings'] = gt_embeddings.astype(np.float32)
            info['vol_points_list'] = vol_points_list.astype(np.float32)
            info['vol_label_list'] = vol_label_list.astype(np.float32)
        else:
            print(image_id)
            info['gt_labels'] = np.array([]).view('long').reshape(-1)
            info['gt_2d_boxes'] = np.array([]).view('float32').reshape(-1,8)
            info['gt_3d_boxes'] = np.array([]).view('float32').reshape(-1,5)
            info['gt_embeddings'] = np.array([]).view('float32').reshape(-1,128)
            info['vol_points_list'] = np.array([]).view('float32').reshape(-1,2048,3)
            info['vol_label_list'] = np.array([]).view('float32').reshape(-1,2048)

        lvs6d_infos.append(info)

    print('{} sample: {}'.format(split_name, len(lvs6d_infos)))
    print(loc_max)
    data = dict(infos=lvs6d_infos)
    info_path = osp.join(root_path, f'lvs6d_infos_{split_name}_debug.pkl')
    mmcv.dump(data, info_path)

if __name__ == "__main__":
    create_lvs6d_infos("./data/lvs6d", "train")
    create_lvs6d_infos("./data/lvs6d", "test")
