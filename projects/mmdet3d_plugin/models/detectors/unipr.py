import copy
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS, builder
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask

from projects.mmdet3d_plugin.models.utils import DiagonalGaussianDistribution, chamfer_distance_numpy

@DETECTORS.register_module()
class Unipr(MVXTwoStageDetector):
    """Unipr"""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_2d_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Unipr, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask

        self.all = [0] * 192
        self.iou = [0] * 192
        self.ape = [0] * 192
        self.acd = [0] * 192
        self.hit = [0] * 192
        
        self.vis_count = 0
        self.inference_single_mode = False

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if isinstance(img, list):
            img = torch.stack(img, dim=0)

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)
            if img.dim() == 5:
                if img.size(0) == 1 and img.size(1) != 1:
                    img.squeeze_()
                else:
                    B, N, C, H, W = img.size()
                    img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        return img_feats

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_parts_3d,
                          gt_embeddings,
                          part_bboxes_3d,
                          part_labels_3d,
                          part_embeddings,
                          gt_vol_points,
                          gt_vol_label,
                          gt_l_masks,
                          gt_r_masks,
                          gt_mask,
                          gt_disp,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, gt_parts_3d, gt_embeddings, part_bboxes_3d, part_labels_3d, part_embeddings, gt_vol_points, gt_vol_label, gt_l_masks, gt_r_masks, gt_mask, gt_disp, outs, img_metas]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        
        return losses

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_2d=None,
                      gt_bboxes_3d=None,
                      gt_labels_2d=None,
                      gt_labels_3d=None,
                      gt_parts_3d=None,
                      gt_embeddings=None,
                      part_bboxes_3d=None,
                      part_labels_3d=None,
                      part_embeddings=None,
                      gt_vol_points=None,
                      gt_vol_label=None,
                      gt_l_masks=None,
                      gt_r_masks=None,
                      gt_mask=None,
                      gt_disp=None,
                      img=None,
                      gt_bboxes_ignore=None,
                      mask=None):
        """Forward training function.
        """
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d, gt_labels_3d, gt_parts_3d, gt_embeddings, part_bboxes_3d, part_labels_3d, part_embeddings, gt_vol_points, gt_vol_label, gt_l_masks, gt_r_masks, gt_mask, gt_disp, img_metas, gt_bboxes_ignore)
        losses.update(losses_pts)

        return losses
  
    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)

    def simple_test_pts(self, x, img_metas, rescale=False, gt_labels_2d=None, gt_bboxes_3d=None, gt_labels_3d=None, gt_parts_3d=None, gt_embeddings=None, part_bboxes_3d=None, part_labels_3d=None, part_embeddings=None, gt_vol_points=None, gt_vol_label=None):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, img_metas)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels, _, _ in bbox_list
        ]
        
        gt_bboxes_3d = gt_bboxes_3d[0][0].tensor.cpu().numpy()
        pred_bboxes_3d = bbox_results[0]['boxes_3d'].tensor.cpu().numpy()
        embeddings = bbox_list[0][4]
        gt_embeddings = gt_embeddings[0][0]

        if len(pred_bboxes_3d) != 0:
            bf = img_metas[0]['intrinsics'][0][0][0] * img_metas[0]['interocular_distance']
            cam_k = img_metas[0]['intrinsics'][0]
            cam_k = np.asarray(cam_k)
            cam_k = np.linalg.inv(cam_k)
            cam_k = cam_k[:3,:3]
            uvd = pred_bboxes_3d[:,:3].copy()
            uvd[:,2] = bf / uvd[:,2]
            uvd[:,:2] = uvd[:,:2] * uvd[:,2:3]
            pred_center = np.dot(cam_k,uvd.T).T
        else:
            pred_center = pred_bboxes_3d[:, :3]

        if self.inference_single_mode:
            return bbox_list[0][1], pred_center, pred_bboxes_3d[:,3:4], embeddings

        density = self.resolution
        gap = 2. / density
        x = np.linspace(-1, 1, density+1)
        y = np.linspace(-1, 1, density+1)
        z = np.linspace(-1, 1, density+1)

        xv, yv, zv = np.meshgrid(x, y, z)
        grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].cuda()
        points = grid.clone().view(density+1,density+1,density+1,3)
        
        for i in range(gt_bboxes_3d.shape[0]):
            self.all[gt_labels_2d[0][0][i].item()] += 1
            mean = gt_embeddings[i, :64].reshape(-1, 1, 64)
            logvar = gt_embeddings[i, 64:].reshape(-1, 1, 64)
            posterior = DiagonalGaussianDistribution(mean, logvar)
            sampled_embedding = posterior.sample()
            
            with torch.no_grad():
                output = self.pts_bbox_head.ae_model.decode_emb(sampled_embedding, grid)['logits']
                volume = output.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
            
                coordinates = np.argwhere(volume > -1)
                point = points[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]]
                gt_point = point.clone().cpu().numpy()
                if len(point) != 0:
                    gt_point *= gt_bboxes_3d[i,3]
                    gt_point += gt_bboxes_3d[i,:3]
                
                for j in range(pred_center.shape[0]):
                    position_error = np.linalg.norm(gt_bboxes_3d[i,:3] - pred_center[j])
                    scale_error = np.linalg.norm(gt_bboxes_3d[i,3] - pred_bboxes_3d[j,3])
                    if position_error <= 4 and scale_error <= 4:
                        self.hit[gt_labels_2d[0][0][i].item()] += 1

                        mean = embeddings[j, :64].reshape(-1, 1, 64)
                        logvar = embeddings[j, 64:].reshape(-1, 1, 64)
                        posterior = DiagonalGaussianDistribution(mean, logvar)
                        sampled_embedding = posterior.sample()
                        with torch.no_grad():
                            output = self.pts_bbox_head.ae_model.decode_emb(sampled_embedding, grid)['logits']
                            volume = output.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                        
                        coordinates = np.argwhere(volume > -2)
                        point = points[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]]
                        pred_point = point.clone().cpu().numpy()
                        
                        if len(point) == 0:
                            continue
                        
                        pred_point *= pred_bboxes_3d[j,3]
                        pred_point += pred_center[j,:3]
                        
                        pred_max = np.amax(pred_point, axis=0)
                        pred_min = np.amin(pred_point, axis=0)
                        gt_max = np.amax(gt_point, axis=0)
                        gt_min = np.amin(gt_point, axis=0)
                    
                        overlap_min = np.maximum(pred_min, gt_min)
                        overlap_max = np.minimum(pred_max, gt_max)

                        # intersections and union
                        if np.amin(overlap_max - overlap_min) < 0:
                            intersections = 0
                        else:
                            intersections = np.prod(overlap_max - overlap_min)
                        union = np.prod(pred_max - pred_min) + \
                            np.prod(gt_max - gt_min) - intersections
                        overlaps = intersections / union

                        CD = chamfer_distance_numpy(torch.from_numpy(pred_point), torch.from_numpy(gt_point)).item()
                        
                        if overlaps > 0.5:
                            self.iou[gt_labels_2d[0][0][i].item()] += 1
                        self.ape[gt_labels_2d[0][0][i].item()] += position_error
                        self.acd[gt_labels_2d[0][0][i].item()] += CD
                        
                        break
        
        vis_num = getattr(self, 'vis_num', 0)
        if vis_num and self.vis_count < vis_num:
            self.vis_count += 1
    
            fig, axes = plt.subplots(1, 3, figsize=(9, 2), dpi=200)

            axes[0].set_title('input')
            axes[1].set_title('gt')
            axes[2].set_title('pred')

            ori_img = img_metas[0]['ori_img'][0]
            ori_img = ori_img[...,[2,1,0]]
            intrinsics = img_metas[0]['intrinsics']

            pil_image = Image.fromarray(np.uint8(ori_img))
            pil_image = pil_image.resize((896, 560))
            ori_img = np.array(pil_image)

            for ax in axes:
                ax.set_xlim(0, 896)
                ax.set_ylim(560, 0)
                ax.axis('off')
                ax.imshow(ori_img/255)
            
            for idx, embedding in enumerate(gt_embeddings):
                mean = embedding[:64].reshape(-1, 1, 64)
                logvar = embedding[64:].reshape(-1, 1, 64)
                posterior = DiagonalGaussianDistribution(mean, logvar)
                sampled_embedding = posterior.sample()
                
                with torch.no_grad():
                    output = self.pts_bbox_head.ae_model.decode_emb(sampled_embedding, grid)['logits']
                    volume = output.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                
                coordinates = np.argwhere(volume > -2)
                point = points[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]]
                space_point = point.clone().cpu().numpy()
                if len(point) != 0:
                    space_point *= gt_bboxes_3d[idx,3]
                    space_point += gt_bboxes_3d[idx,:3]

                    point = np.dot(intrinsics[0][:3, :3], space_point.T[:]) 
                    point = point.T[np.argsort(-point.T[:,2])].T
                    depth = point[-1]
                    point = point[:2] / point[-1]
                    axes[1].scatter(point[0], point[1], s=0.1, c=depth, cmap='rainbow', vmin=depth.min(), vmax=depth.max(), alpha = 1.0)

            for idx, embedding in enumerate(embeddings):
                mean = embedding[:64].reshape(-1, 1, 64)
                logvar = embedding[64:].reshape(-1, 1, 64)
                posterior = DiagonalGaussianDistribution(mean, logvar)
                sampled_embedding = posterior.sample()
                
                with torch.no_grad():
                    output = self.pts_bbox_head.ae_model.decode_emb(sampled_embedding, grid)['logits']
                    volume = output.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
                
                coordinates = np.argwhere(volume > -2)
                point = points[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]]
                space_point = point.clone().cpu().numpy()
                if len(point) != 0:
                    space_point *= pred_bboxes_3d[idx,3]
                    space_point += pred_center[idx,:3]

                    point = np.dot(intrinsics[0][:3, :3], space_point.T[:]) 
                    point = point.T[np.argsort(-point.T[:,2])].T
                    depth = point[-1]
                    point = point[:2] / point[-1]
                    axes[2].scatter(point[0], point[1], s=0.1, c=depth, cmap='rainbow', vmin=depth.min(), vmax=depth.max(), alpha = 1.0)

                # import mcubes
                # import trimesh
                # verts, faces = mcubes.marching_cubes(volume, -2)
                # verts *= gap
                # verts -= 1.
                # m = trimesh.Trimesh(verts, faces)
                # m.export(f'copass_mono/val{idx}.glb')

            work_dir = getattr(self, 'work_dir', 'vis_output')
            vis_dir = os.path.join(work_dir, "vis_output")
            os.makedirs(vis_dir, exist_ok=True)
            fig.savefig(f"{vis_dir}/vis_{self.vis_count}.png", bbox_inches='tight', pad_inches=0.02)
            print(f"Saved visualization to {vis_dir}/vis_{self.vis_count}.png")
            if self.vis_count >= vis_num:
                print(f"Generated {vis_num} visualization images.")

        return bbox_results

    
    def simple_test(self, img_metas, img=None, rescale=False, gt_bboxes_2d=None, gt_bboxes_3d=None, gt_labels_2d=None, gt_labels_3d=None, gt_parts_3d=None, 
                    gt_embeddings=None, part_bboxes_3d=None, part_labels_3d=None, part_embeddings=None, gt_vol_points=None, gt_vol_label=None, **kwargs):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        
        bbox_pts = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale, gt_labels_2d=gt_labels_2d, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, gt_parts_3d=gt_parts_3d, 
            gt_embeddings=gt_embeddings, part_bboxes_3d=part_bboxes_3d, part_labels_3d=part_labels_3d, part_embeddings=part_embeddings, gt_vol_points=gt_vol_points, gt_vol_label=gt_vol_label)
        
        return bbox_pts

    def aug_test_pts(self, feats, img_metas, rescale=False):
        feats_list = []
        for j in range(len(feats[0])):
            feats_list_level = []
            for i in range(len(feats)):
                feats_list_level.append(feats[i][j])
            feats_list.append(torch.stack(feats_list_level, -1).mean(-1))
        outs = self.pts_bbox_head(feats_list, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats = self.extract_feats(img_metas, imgs)
        img_metas = img_metas[0]
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.aug_test_pts(img_feats, img_metas, rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list
    