import torch 
from .array_converter import array_converter

@array_converter(apply_to=('points', 'cam2img'))
def points_img2cam(points, cam2img):
    """Project points in image coordinates to camera coordinates.

    Args:
        points (torch.Tensor): 2.5D points in 2D images, [N, 3],
            3 corresponds with x, y in the image and depth.
        cam2img (torch.Tensor): Camera intrinsic matrix. The shape can be
            [3, 3], [3, 4] or [4, 4].

    Returns:
        torch.Tensor: points in 3D space. [N, 3],
            3 corresponds with x, y, z in 3D space.
    """
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[:cam2img.shape[0], :cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    # Do operation in homogeneous coordinates.
    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]

    return points3D

def normalize_bbox(bboxes, pc_range):
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    scale = bboxes[..., 3:4].log()
    scale_2d = bboxes[..., 4:5]
    
    normalized_bboxes = torch.cat([cx, cy, cz, scale, scale_2d], dim=-1)

    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 2:3]

    # size
    scale = normalized_bboxes[..., 3:4]
    scale = scale.exp()
    scale_2d = normalized_bboxes[..., 4:5]

    denormalized_bboxes = torch.cat([cx, cy, cz, scale, scale_2d], dim=-1)

    return denormalized_bboxes

def cross_product(u, v):
    i = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    j = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    k = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    out = torch.cat((i.unsqueeze(-1), j.unsqueeze(-1), k.unsqueeze(-1)), -1)  # bx3
    return out


def bbox_cxcywh_to_xyxy_s(bbox):
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    if bbox.shape[-1] == 4:
        cx, cy, w, h = bbox.split((1, 1, 1, 1), dim=-1)
        bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    else:
        cx, cy, w, h, ca, cb, wa, wb = bbox.split((1, 1, 1, 1, 1, 1, 1, 1), dim=-1)
        bbox_new = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h), (ca - 0.5 * wa), (cb - 0.5 * wb), (ca + 0.5 * wa), (cb + 0.5 * wb)]
    return torch.cat(bbox_new, dim=-1)


def bbox_xyxy_to_cxcywh_s(bbox):
    """Convert bbox coordinates from (x1, y1, x2, y2) to (cx, cy, w, h).

    Args:
        bbox (Tensor): Shape (n, 4) for bboxes.

    Returns:
        Tensor: Converted bboxes.
    """
    if bbox.shape[-1] == 4:
        x1, y1, x2, y2 = bbox.split((1, 1, 1, 1), dim=-1)
        bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)]
    else:
        x1, y1, x2, y2, a1, b1, a2, b2 = bbox.split((1, 1, 1, 1), dim=-1)
        bbox_new = [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1), (a1 + a2) / 2, (b1 + b2) / 2, (a2 - a1), (b2 - b1)]
    return torch.cat(bbox_new, dim=-1)