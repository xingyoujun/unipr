import os
import time
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from copy import deepcopy
import trimesh
import shutil
import mcubes
import trimesh
import torch

from mmcv.parallel import collate, scatter
from mmdet3d.apis import init_model
from mmdet3d.core import (Box3DMode, LiDARInstance3DBoxes)
from mmdet3d.datasets.pipelines import Compose

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for UNIPR')
    parser.add_argument('--config', type=str, default='./projects/configs/unipr/unipr_dinov2_uvd.py', help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, default='./ckpts/unipr.pth', help='Path to the model checkpoint')
    parser.add_argument('--resolution', type=int, default=64, help='Resolution for the ae model')
    parser.add_argument('--image_left', type=str, default='./assets/real_left.jpg', help='Path to the left image')
    parser.add_argument('--image_right', type=str, default='./assets/real_right.jpg', help='Path to the right image')
    parser.add_argument('--interocular_distance', type=float, default=13.0, help='Interocular distance')
    parser.add_argument('--cam_intrinsic', type=str, default='1437.19523514,0,955.62555885,0,1437.19523514,592.37124634,0,0,1', help='Camera intrinsic matrix flattened (9 values separated by comma)')
    return parser.parse_args()

args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import importlib
plugin_dir = 'projects/mmdet3d_plugin/'
_module_dir = os.path.dirname(plugin_dir)
_module_dir = _module_dir.split('/')
_module_path = _module_dir[0]
for m in _module_dir[1:]:
    _module_path = _module_path + '.' + m

plg_lib = importlib.import_module(_module_path)

from projects.mmdet3d_plugin.models.utils import DiagonalGaussianDistribution

config = args.config
checkpoint = args.checkpoint

model = init_model(config, checkpoint, device=device)

config_name = os.path.splitext(os.path.basename(config))[0]
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join("inference_results", "single_mode", config_name , timestamp)
os.makedirs(output_dir, exist_ok=True)
mesh_dir = os.path.join(output_dir, "mesh")
os.makedirs(mesh_dir, exist_ok=True)
print(f"Create Output Dir: {output_dir}")

cfg = model.cfg
device = next(model.parameters()).device  # model device
test_pipeline = deepcopy(cfg.data.test.pipeline)
test_pipeline = Compose(test_pipeline)
box_type_3d = LiDARInstance3DBoxes
box_mode_3d = Box3DMode.LIDAR
input_dict = dict(
    box_type_3d=box_type_3d,
    box_mode_3d=box_mode_3d,
    img_fields=[],
    bbox3d_fields=[],
    pts_mask_fields=[],
    pts_seg_fields=[],
    bbox_fields=[],
    mask_fields=[],
    seg_fields=[])

image_paths = [args.image_left, args.image_right]

intrinsic_vals = [float(x) for x in args.cam_intrinsic.split(',')]
cam_intrinsic = np.array(intrinsic_vals).reshape(3,3)

intrinsic = cam_intrinsic
viewpad = np.eye(4)
viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
rightview = viewpad.copy()
intrinsics = [viewpad, rightview]

input_dict.update(
    image_id=0,
    interocular_distance=args.interocular_distance,
    img_filename=image_paths,
    intrinsics=intrinsics,
    ann_info=dict(
        gt_bboxes_2d=[],
        gt_labels_2d=[],
        gt_bboxes_3d=LiDARInstance3DBoxes([],
            box_dim=10,
            origin=(0.5, 0.5, 0.5)).convert_to(box_mode_3d),
        gt_labels_3d=[],
        gt_parts_3d=[],
        gt_vol_points=[],
        gt_vol_label=[],
        gt_embeddings=[],
        gt_voxels=[],
        part_embeddings=[],)
)

data = test_pipeline(input_dict)

data = collate([data], samples_per_gpu=1)
if next(model.parameters()).is_cuda:
    data = scatter(data, [device.index])[0]
else:
    data['img_metas'] = data['img_metas'][0].data
    data['img'] = data['img'][0].data

model.inference_single_mode = True
model.resolution = args.resolution

# forward the model
with torch.no_grad():
    scores, pred_center, pred_dim, pred_embbeding = model(return_loss=False, rescale=True, **data)

density = args.resolution
gap = 2. / density
x = np.linspace(-1, 1, density+1)
y = np.linspace(-1, 1, density+1)
z = np.linspace(-1, 1, density+1)

xv, yv, zv = np.meshgrid(x, y, z)
grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].cuda()
points = grid.clone().view(density+1,density+1,density+1,3)

Width, Height = 896, 560
mydpi=200
fig = plt.figure(figsize=(Width/mydpi,Height/mydpi),dpi=mydpi)
plt.axis('off')
plt.xlim(0,Width)
plt.ylim(Height,0)
ori_imgs = data['img_metas'][0][0]['ori_img']
ori_img = ori_imgs[0][...,[2,1,0]]
pil_image = Image.fromarray(np.uint8(ori_img))
pil_image = pil_image.resize((Width, Height))
ori_img = np.array(pil_image)
plt.imshow(ori_img/255)

intrinsics = data['img_metas'][0][0]['intrinsics']

for i, src_path in enumerate(image_paths):
    side = "left" if i == 0 else "right"
    ext = os.path.splitext(src_path)[1]
    dst_path = os.path.join(output_dir, f"input_{side}{ext}")
    shutil.copy(src_path, dst_path)
    print(f"Copy Source Input Images.")

for idx, embedding in enumerate(pred_embbeding):
    mean = embedding[:64].reshape(-1, 1, 64)
    logvar = embedding[64:].reshape(-1, 1, 64)
    posterior = DiagonalGaussianDistribution(mean, logvar)
    sampled_embedding = posterior.sample()
    
    with torch.no_grad():
        output = model.pts_bbox_head.ae_model.decode_emb(sampled_embedding, grid)['logits']
        volume = output.view(density+1, density+1, density+1).permute(1, 0, 2).cpu().numpy()
    
    coordinates = np.argwhere(volume > -2)
    point = points[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]]
    space_point = point.clone().cpu().numpy()
    if len(point) != 0:
        space_point *= pred_dim[idx]
        space_point += pred_center[idx,:3]

        point = np.dot(intrinsics[0][:3, :3], space_point.T[:]) 
        point = point.T[np.argsort(-point.T[:,2])].T
        depth = point[-1]
        point = point[:2] / point[-1]
        plt.scatter(point[0], point[1], s=0.1, c=depth, cmap='rainbow', vmin=depth.min(), vmax=depth.max(), alpha = 1.0)

    verts, faces = mcubes.marching_cubes(volume, -2)
    verts *= gap
    verts -= 1.
    m = trimesh.Trimesh(verts, faces)
    obj_filename = f"object_{idx:03d}.glb"
    obj_path = os.path.join(mesh_dir, obj_filename)
    m.export(obj_path)

vis_path = os.path.join(output_dir, 'vis.png')
plt.savefig(vis_path)