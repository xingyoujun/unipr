<p align="center">
  <h1 align="center">UniPR: Unified Object-level Real-to-Sim Perception and Reconstruction from a Single Stereo Pair</h1>
  <p align="center">
    <a href="https://xingyoujun.github.io/">Chuanrui Zhang</a>
    &nbsp;·&nbsp;
    <a href="https://heiheishuang.xyz/">Yingshuang Zou</a>
    &nbsp;·&nbsp;
    <a>Zhengxian Wu</a>
    &nbsp;·&nbsp;
    <a href="https://ygling2008.github.io/">Yonggen Ling</a>
    &nbsp;·&nbsp;
    <a>Yuxiao Yang</a>
    &nbsp;·&nbsp;
    <a href="https://ziweiwangthu.github.io/">Ziwei Wang</a>
  </p>
  <h3 align="center">CVPR 2026</h3>
  <h3 align="center">
    <a href="https://arxiv.org/abs/2603.19616">Paper</a> | 
    <a href="https://xingyoujun.github.io/unipr/">Project Page</a> | 
    <a href="https://huggingface.co/xingyoujun/unipr">Pretrained Models</a> | 
    <a href="https://huggingface.co/datasets/xingyoujun/lvs6d">Dataset</a>
  </h3>
</p>

## Installation

> **⚠️ Important Note:** We strongly recommend **strictly following** the exact versions of Python, PyTorch, MMCV, and mmdet3d specified below. Other versions have not been tested and will likely cause compilation failures or runtime errors due to library incompatibilities. Furthermore, local CUDA versions ranging from 11.1 to 11.8 have been successfully tested and are fully compatible with the exact same installation commands provided below.

**a. Create a conda virtual environment and activate it.**

```bash
conda create --name unipr -y python=3.8
conda activate unipr
```

**b. Install PyTorch and torchvision.**

```bash
# Please ensure the PyTorch version matches your CUDA environment
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install mmcv and mmdet3d.**

```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.24.1
pip install mmsegmentation==0.20.2

git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 
pip install -e .
cd ..
```

**d. Install other requirements.**

```bash
pip install trimesh==4.0.0
pip install einops
pip install setuptools==58.0.4
pip install yapf==0.40.1
pip install timm==1.0.9
pip install PyMCubes==0.1.4

pip install wandb
```

## Acquiring Datasets

### LVS6D Dataset

We construct a large-vocabulary stereo dataset, **LVS6D**, comprising over 6,300 objects for training and evaluation. You can download the dataset files from our [Hugging Face repository](https://huggingface.co/datasets/xingyoujun/lvs6d).

Please download `category_info.json`, `lvs6d_test.zip`, and all 10 parts of the training set (`lvs6d_train_part.tar.gz.00` through `09`). Merge and extract the compressed files, and place everything into the `data/lvs6d/` directory.

```bash
# Extract the test set
unzip lvs6d_test.zip

# Merge and extract the multi-part training set
cat lvs6d_train_part.tar.gz.* > lvs6d_train.tar.gz
tar -xzvf lvs6d_train.tar.gz
```

#### Data Structure
After extraction and preparation, your directory structure should look exactly like this:

```text
data/lvs6d/
├── lvs6d_train/
├── lvs6d_test/
├── category_info.json
├── latents_train.npz (optional)
├── latents_test.npz (optional)
├── lvs6d_pc_center.npz (optional)
├── lvs6d_scale.npz (optional)
├── lvs6d_infos_train.pkl
├── lvs6d_infos_test.pkl
├── lvs6d_infos_train_debug.pkl (optional)
└── lvs6d_infos_test_debug.pkl (optional)
```

#### Data Preparation

> **⚠️ Important Note:** We strongly recommend directly downloading the pre-processed annotation `.pkl` files (`lvs6d_infos_train.pkl`, `lvs6d_infos_test.pkl`, and their debug versions) from our Hugging Face repository. 
> 
> Generating these `.pkl` files from scratch using our data converter scripts is **currently not supported**, as the required raw object voxel annotations have not been uploaded yet. 
> 
> As long as you have downloaded the `.pkl` files, the LVS6D datasets, and `category_info.json` into the correct directory structure, you are completely set up. You can proceed directly to training and testing without any further data generation steps.

## Running the Code

### Pre-trained Models

Before running inference or training, please download all the required pre-trained weights from our [Hugging Face repository](https://huggingface.co/xingyoujun/unipr) and place them in the `./ckpts/` directory.

#### Directory Structure
After downloading, your `./ckpts/` directory should be organized exactly like this:

```text
ckpts/
├── dinov2_vitb14_reg4_pretrain.npz
├── shape_ae.pth
└── unipr.pth
```

#### Weight Descriptions
* **`unipr.pth`**: The main pre-trained UniPR model weights used for inference and evaluation on the LVS6D dataset.
* **`shape_ae.pth`**: The pre-trained shape autoencoder weights, utilized by the reconstruction module for object-level shape prior and decoding. 
* **`dinov2_vitb14_reg4_pretrain.npz`**: Pre-trained weights for the DINOv2 backbone. Because our framework strictly relies on PyTorch < 2.0 (v1.9.0), the official PyTorch 2.0+ DINOv2 checkpoints are incompatible. We converted them to `.npz` format for stable loading during the pre-training initialization phase.

Please ensure this entire folder is placed at the specified path so the model modules can load the checkpoints correctly.

### Evaluation

#### Quick Start: Single-mode Inference

You can easily run the inference test using our provided shell script:

```bash
sh infer.sh
```

**Inside `infer.sh`:**
This script provides a ready-to-use pipeline. It passes default arguments including the configuration, model checkpoints, SDF path, and reconstruction resolution. Most importantly, it allows you to specify the input stereo images (`image_left` and `image_right`) along with their corresponding camera parameters (`interocular_distance` and `cam_intrinsic`):

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --resolution "$RESOLUTION" \  # Default 64 for visulization
    --image_left "$IMAGE_LEFT" \
    --image_right "$IMAGE_RIGHT" \
    --interocular_distance "$INTEROCULAR_DISTANCE" \
    --cam_intrinsic "$CAM_INTRINSIC"
```

> **⚠️ Important Note:** Our model is trained specifically on data with fixed camera resolutions and baseline distances. For testing, please use images and parameters consistent with the provided examples in the `assets/` directory. Directly testing on images from other stereo camera setups has not been verified and will likely require fine-tuning the model on a custom dataset.

*The rendered 3D meshes (`.glb` files), original input images, and 2D visualization results will be stored under `inference_results/single_mode/` with corresponding timestamps.*

#### Quantitative Evaluation on LVS6D
To evaluate the model's performance on the LVS6D dataset and calculate metrics, you can run our evaluation script:

```bash
sh metric.sh
```

Inside `metric.sh`, we use `tools/dist_test.sh` to perform the testing. By default, it runs on a single GPU using the DINOv2 backbone:

```bash
CUDA_VISIBLE_DEVICES=2 tools/dist_test.sh projects/configs/unipr/unipr_dinov2_uvd.py ./ckpts/unipr.pth 1 28501 --eval bbox --resolution 32 --vis 5
```

**Command Breakdown:**
* `CUDA_VISIBLE_DEVICES=2`: Specifies the physical GPU ID to use.
* `1`: Specifies the total number of GPUs used for testing.
* `28501`: The port assigned for the distributed testing process.
* `--eval bbox`: Instructs the script to compute evaluation metrics for 3D bounding boxes.
* **`--resolution 32`**: Sets the resolution for the 3D implicit reconstruction evaluation. Please note that a higher resolution yields finer 3D meshes but will significantly slow down the testing process. All quantitative results reported in our paper were obtained using this default resolution of 32.
* `--vis 5`: Enables visualization during testing (e.g., saving visualization results for qualitative analysis).

**Evaluation Outputs:**
After the evaluation is complete, the quantitative results will be saved as a CSV file in a timestamped directory. For example: 
`inference_results/unipr_dinov2_uvd/20260324_200837/metrics_output.csv`

The generated CSV report features a two-tier structure:
1. **Difficulty-Level Summary:** The first three rows (`SUMMARY-easy`, `SUMMARY-Medium`, `SUMMARY-hard`) provide aggregated metrics averaged across three specific difficulty levels (1: Easy, 2: Medium, 3: Hard), giving a clear overview of the model's robustness.
2. **Category-Level Details:** The subsequent rows provide the detailed, per-class evaluation metrics for all objects in the vocabulary (e.g., `anise`, `antique`, `apple`).

The key headers in the report represent the following metrics:
* **`num`**: The total number of ground truth object instances for that specific category or difficulty level.
* **`hit`**: The number of correctly matched/localized object predictions.
* **`ap-iou50`**: 3D Average Precision (or Accuracy) calculated at an Intersection over Union (IoU) threshold of 0.50.
* **`ape`**: Average Pose Error, measuring the alignment accuracy of the estimated location.
* **`acd`**: Average Chamfer Distance, evaluating the quality and geometric accuracy of the reconstructed 3D object shape.

### Training

To train UniPR from scratch, run the following command using our script:

```bash
sh train.sh
```

Inside `train.sh`, we use `tools/dist_train.sh` for distributed training. You can specify the visible GPUs, the configuration file, the number of GPUs, and the working directory. 

By default, the script trains the model with the **DINOv2** backbone using 1 GPU:

```bash
# For DINOv2 backbone
CUDA_VISIBLE_DEVICES=0 tools/dist_train.sh projects/configs/unipr/unipr_dinov2_uvd.py 1 --work-dir work_dirs/unipr/
```

> **Note:** Please adjust `CUDA_VISIBLE_DEVICES` and the GPU count parameter (e.g., `1`) based on your available hardware resources. If you are using multiple GPUs, ensure the GPU count matches the number of devices specified in `CUDA_VISIBLE_DEVICES`.

## BibTeX

If you find our work helpful, please consider citing:

```bibtex
@article{zhang2026unipr,
  title={UniPR: Unified Object-level Real-to-Sim Perception and Reconstruction from a Single Stereo Pair},
  author={Zhang, Chuanrui and Zou, Yingshuang and Wu, ZhengXian and Ling, Yonggen and Yang, Yuxiao and Wang, Ziwei},
  journal={arXiv preprint arXiv:2603.19616},
  year={2026}
}
```

## Acknowledgements

This project is heavily inspired by and built upon several excellent open-source codebases. We would like to specifically thank the authors and maintainers of [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) for their foundational transformer architectures and 3D detection frameworks. Furthermore, we have incorporated concepts, code snippets, and pre-trained models from [DINOv2](https://github.com/facebookresearch/dinov2) and [3DShape2VecSet](https://github.com/1zb/3DShape2VecSet).

Many thanks to all these projects for their outstanding contributions to the open-source community!