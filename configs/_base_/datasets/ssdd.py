_base_ = [
    '../default_runtime.py',
    '../schedules/schedule_1x.py',
]
# _base_ = [
#     '../../s2anet/s2anet_r50_fpn_1x_dota_le135.py',
# ]

dataset_type = 'SARDataset'
data_root = 'data/ssdd/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)  # 图像标准化配置
train_pipeline = [
    # 训练数据预处理流程
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(type='LoadAnnotations', with_bbox=True),  # 从文件加载标注信息
    dict(type='RResize', img_scale=(608, 608)),  # 图像 Resize 预处理
    dict(type='RRandomFlip', flip_ratio=0.5),  # 随机翻转
    dict(type='Normalize', **img_norm_cfg),  # 图像标准化
    dict(type='Pad', size_divisor=32),  # 图像填充
    dict(type='DefaultFormatBundle'),  # 默认的格式转换
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])  # 数据集收集
]
test_pipeline = [
    # 测试数据预处理流程
    dict(type='LoadImageFromFile'),  # 从文件加载图像
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])  # 多尺度翻转增强
]
data = dict(
    samples_per_gpu=2,  # 每个 GPU 的样本数
    workers_per_gpu=2,  # 每个 GPU 的数据加载器数
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/labelTxt/',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/inshore/labelTxt/',
        img_prefix=data_root + 'test/inshore/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/offshore/labelTxt/',
        img_prefix=data_root + 'test/offshore/images/',
        pipeline=test_pipeline))

model = dict(
    type='RotatedRetinaNet',  # 检测器(detector)名称
    backbone=dict(  # 主干网络的配置文件
        type='ResNet',  # # 主干网络的类别
        depth=50,  # 主干网络的深度
        num_stages=4,  # 主干网络阶段(stages)的数目
        out_indices=(0, 1, 2, 3),  # 每个阶段产生的特征图输出的索引
        frozen_stages=1,  # 第一个阶段的权重被冻结
        zero_init_residual=False,  # 是否对残差块(resblocks)中的最后一个归一化层使用零初始化(zero init)让它们表现为自身
        norm_cfg=dict(  # 归一化层(norm layer)的配置项
            type='BN',  # 归一化层的类别，通常是 BN 或 GN
            requires_grad=True),  # 是否训练归一化里的 gamma 和 beta
        norm_eval=True,  # 是否冻结 BN 里的统计项
        style='pytorch',  # 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积， 'caffe' 意思是步长为2的层为 1x1 卷积。
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),  # 加载通过 ImageNet 预训练的模型
    neck=dict(
        type='FPN',  # 检测器的 neck 是 FPN， 我们同样支持 'ReFPN'
        in_channels=[256, 512, 1024, 2048],  # 输入通道数，这与主干网络的输出通道一致
        out_channels=256,  # 金字塔特征图每一层的输出通道
        start_level=1,  # 用于构建特征金字塔的主干网络起始输入层索引值
        add_extra_convs='on_input',  # 决定是否在原始特征图之上添加卷积层
        num_outs=5),  # 决定输出多少个尺度的特征图(scales)
    bbox_head=dict(
        type='RotatedRetinaHead',# bbox_head 的类型是 'RRetinaHead'
        num_classes=1,  # 分类的类别数量
        in_channels=256,  # bbox head 输入通道数
        stacked_convs=4,  # head 卷积层的层数
        feat_channels=256,  # head 卷积层的特征通道
        assign_by_circumhbbox='oc',  # obb2hbb 的旋转定义方式
        anchor_generator=dict(  # 锚点(Anchor)生成器的配置
            type='RotatedAnchorGenerator',  # 锚点生成器类别
            octave_base_scale=4,  # RetinaNet 用于生成锚点的超参数，特征图 anchor 的基本尺度。值越大，所有 anchor 的尺度都会变大。
            scales_per_octave=3,  #  RetinaNet 用于生成锚点的超参数，每个特征图有3个尺度
            ratios=[1.0, 0.5, 2.0],  # 高度和宽度之间的比率
            strides=[8, 16, 32, 64, 128]),  # 锚生成器的步幅。这与 FPN 特征步幅一致。如果未设置 base_sizes，则当前步幅值将被视为 base_sizes。
        bbox_coder=dict(  # 在训练和测试期间对框进行编码和解码
            type='DeltaXYWHAOBBoxCoder',  # 框编码器的类别
            angle_range='oc',  # 框编码器的旋转定义方式
            norm_factor=None,  # 框编码器的范数
            edge_swap=False,  # 设置是否启用框编码器的边缘交换
            proj_xy=False,  # 设置是否启用框编码器的投影
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),  # 用于编码和解码框的目标均值
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),  # 用于编码和解码框的标准差
        loss_cls=dict(  # 分类分支的损失函数配置
            type='FocalLoss',  # 分类分支的损失函数类型
            use_sigmoid=True,  #  是否使用 sigmoid
            gamma=2.0,  # Focal Loss 用于解决难易不均衡的参数 gamma
            alpha=0.25,  # Focal Loss 用于解决样本数量不均衡的参数 alpha
            loss_weight=1.0),  # 分类分支的损失权重
        loss_bbox=dict(  # 回归分支的损失函数配置
            type='L1Loss',  # 回归分支的损失类型
            loss_weight=1.0)),  # 回归分支的损失权重
    train_cfg=dict(  # 训练超参数的配置
        assigner=dict(  # 分配器(assigner)的配置
            type='MaxIoUAssigner',  # 分配器的类型
            pos_iou_thr=0.5,  # IoU >= 0.5(阈值) 被视为正样本
            neg_iou_thr=0.4,  # IoU < 0.4(阈值) 被视为负样本
            min_pos_iou=0,  # 将框作为正样本的最小 IoU 阈值
            ignore_iof_thr=-1,  # 忽略 bbox 的 IoF 阈值
            iou_calculator=dict(type='RBboxOverlaps2D')),  # IoU 的计算器类型
        allowed_border=-1,  # 填充有效锚点(anchor)后允许的边框
        pos_weight=-1,  # 训练期间正样本的权重
        debug=False),  # 是否设置调试(debug)模式
    test_cfg=dict(  # 测试超参数的配置
        nms_pre=2000,  # NMS 前的 box 数
        min_bbox_size=0,  # box 允许的最小尺寸
        score_thr=0.05,  # bbox 的分数阈值
        nms=dict(iou_thr=0.1), # NMS 的阈值
        max_per_img=2000))  # 每张图像的最大检测次数

optimizer_config = dict(  # optimizer hook 的配置文件
    grad_clip=dict(
        max_norm=35,
        norm_type=2))
