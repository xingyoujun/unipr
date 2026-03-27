#!/bin/bash

# Default arguments

CONFIG="./projects/configs/unipr/unipr_dinov2_uvd.py"
CHECKPOINT="./ckpts/unipr.pth"

RESOLUTION=64
IMAGE_LEFT="./assets/unipr_demo_left.png"
IMAGE_RIGHT="./assets/unipr_demo_right.png"
INTEROCULAR_DISTANCE=13.0
CAM_INTRINSIC="1437.19523514,0,955.62555885,0,1437.19523514,592.37124634,0,0,1"

# Run inference
CUDA_VISIBLE_DEVICES=2 python inference.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --resolution "$RESOLUTION" \
    --image_left "$IMAGE_LEFT" \
    --image_right "$IMAGE_RIGHT" \
    --interocular_distance "$INTEROCULAR_DISTANCE" \
    --cam_intrinsic "$CAM_INTRINSIC"
