
_base_ = []
data_mode = 'topdown'
test_pipeline = [
    dict(type='LoadImageFromNDArray'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=(192, 256)),
    dict(type='PackPoseInputs')
]
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(type='Pretrained', prefix='backbone.', checkpoint='')), 
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=26,
        input_size=(192, 256),
        in_featuremap_size=(8, 6),
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(hidden_dims=256, s=128, expansion_factor=2, dropout_rate=0.0, drop_path=0.0, act_fn='SiLU', use_rel_bias=False, pos_enc=False),
        loss=dict(type='KLDiscretLoss', use_target_weight=True, beta=10.0, label_softmax=True),
        decoder=dict(type='SimCCLabel', input_size=(192, 256), sigma=6.0, simcc_split_ratio=2.0, normalize=False, use_dark=False)),
    test_cfg=dict(flip_test=False) 
)
dataset_meta_info = dict(dataset_name='halpe26', keypoint_info={})
