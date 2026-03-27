_base_ = [
    '../default_runtime.py'
]
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

point_cloud_range = [0, 0, 64, 960, 600, 256]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = ["anything"]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

train_pkl_ = 'lvs6d_infos_train.pkl'
val_pkl_ = 'lvs6d_infos_test.pkl'
test_pkl_ = 'lvs6d_infos_test.pkl' 

use_stereo = True
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 2
num_points_in_pillar = [32,24,32]
num_points = [32,24,32]
tpv_u_ = 48
tpv_v_ = 32
tpv_d_ = 64

model = dict(
    type='Unipr',
    use_grid_mask=True,
    img_backbone=dict(
        type='DINOV2',
        load_path='./ckpts/dinov2_vitb14_reg4_pretrain.npz',
        out_indices=[8,11],
        out_channels=[384, 768]),
    img_neck=dict(
        type='CPFPN',  ###remove unused parameters 
        in_channels=[384, 768],
        out_channels=_dim_,
        num_outs=_num_levels_),
    pts_bbox_head=dict(
        type='UniprHead',
        load_ae_path="./ckpts/shape_ae.pth",
        num_classes=1,
        in_channels=_dim_,
        num_query=150,
        num_reg_fcs=2,
        code_size=5,
        tpv_u = tpv_u_,
        tpv_v = tpv_v_,
        tpv_d = tpv_d_,
        use_stereo=use_stereo,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='UniprTransformer',
            num_feature_levels=_num_levels_,
            embed_dims=_dim_,
            encoder=dict(
                type='UniprTransformerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                tpv_u = tpv_u_,
                tpv_v = tpv_v_,
                tpv_d = tpv_d_,
                num_points_in_pillar=num_points_in_pillar,
                num_points_in_pillar_cross_view=[8,8,8],
                return_intermediate=False,
                transformerlayers=dict(
                    type='UniprTransformerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='UniprSelfAttention',
                            embed_dims=_dim_),
                        dict(
                            type='SpatialCrossAttention',
                            tpv_u = tpv_u_,
                            tpv_v = tpv_v_,
                            tpv_d = tpv_d_,
                            pc_range=point_cloud_range,
                            deformable_attention=dict(
                                type='TPVMSDeformableAttention3D',
                                embed_dims=_dim_,
                                num_heads=8,
                                num_points=num_points,
                                num_z_anchors=num_points_in_pillar,
                                num_levels=_num_levels_,
                                floor_sampling_offset=False,
                                tpv_u = tpv_u_,
                                tpv_v = tpv_v_,
                                tpv_d = tpv_d_,),
                            embed_dims=_dim_)
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    ffn_emb=_dim_,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                return_intermediate=True,
                num_layers=3,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=3),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    ffn_emb=_dim_,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[0, 0, 64, 960, 600, 256],
            pc_range=point_cloud_range,
            max_num=20,
            voxel_size=voxel_size,
            score_threshold=0.6,
            num_classes=1), 
        positional_encoding=dict(
            type='CustomPositionalEncoding',
            num_feats=_pos_dim_,
            h=tpv_v_,
            w=tpv_u_,
            z=tpv_d_
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=4.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.06),
        loss_embedding=dict(type='KLLoss', loss_weight=0.001),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=4.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.06),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

dataset_type = 'Lvs6dDataset'
data_root = 'data/lvs6d/'

file_client_args = dict(backend='disk')

ida_aug_conf = {
        "resize_lim": (0.44, 0.50),
        "final_dim": (560, 896),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 1200,
        "W": 1920,
        "rand_flip": False,
    }

train_pipeline = [
    dict(type='LoadStereoImageFromFiles', to_float32=True, use_stereo=use_stereo),
    dict(type='LoadCustomAnnotations3D', with_bbox_2d=True, with_bbox_3d=True, with_label_3d=True, with_voxel=True, with_embedding=True, use_stereo=use_stereo),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14),
    dict(type='CustomFormatBundle3D'),
    dict(type='CustomCollect3D', keys=['gt_labels_3d', 'gt_bboxes_3d', 'gt_embeddings', 'gt_vol_points', 'gt_vol_label', 'img'])
]
test_pipeline = [
    dict(type='LoadStereoImageFromFiles', to_float32=True, use_stereo=use_stereo),
    dict(type='LoadCustomAnnotations3D', with_label_2d=True, with_bbox_2d=True, with_bbox_3d=True, with_label_3d=True, with_voxel=True, with_embedding=True, use_stereo=use_stereo),
    dict(type='ResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=14),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='CustomFormatBundle3D'),
            dict(type='CustomCollect3D', keys=['gt_labels_2d', 'gt_labels_3d', 'gt_bboxes_3d', 'gt_embeddings', 'gt_vol_points', 'gt_vol_label', 'img'])
        ])
]
        
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + train_pkl_,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type, pipeline=test_pipeline, data_root=data_root, ann_file=data_root + val_pkl_, classes=class_names, modality=input_modality),
    test=dict(type=dataset_type, pipeline=test_pipeline, data_root=data_root, ann_file=data_root + val_pkl_, classes=class_names, modality=input_modality))

optimizer = dict(
    type='AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.5),
        }),
    weight_decay=0.01)

# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512., grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=5e-2,
    # by_epoch=False
    )
total_epochs = 24
evaluation = dict(interval=25, pipeline=test_pipeline)
find_unused_parameters = False

checkpoint_config = dict(interval=6, max_keep_ckpts=2)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
