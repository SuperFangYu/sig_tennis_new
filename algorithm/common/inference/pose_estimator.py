"""
姿态估计模块 - 封装 RTMPose (MMPose) 用于人体关键点识别
"""
import sys
from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
import torch
from unittest.mock import MagicMock
import warnings

# 屏蔽警告
warnings.filterwarnings("ignore", category=FutureWarning)

# ================================================================
# 1. 兼容性补丁和依赖处理
# ================================================================

# torch.load 兼容性补丁
_original_torch_load = torch.load


def safe_torch_load(*args, **kwargs):
    """安全的 torch.load，处理新版本 PyTorch 的 weights_only 参数"""
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)


torch.load = safe_torch_load

# mmdet 依赖处理（如果不存在则 mock）
try:
    if "mmdet" not in sys.modules:
        mock_mmdet = MagicMock()
        sys.modules["mmdet"] = mock_mmdet
        sys.modules["mmdet.utils"] = mock_mmdet
        sys.modules["mmdet.models"] = mock_mmdet
        sys.modules["mmdet.models.utils"] = mock_mmdet
    import mmcv

    if not hasattr(mmcv, 'ops'):
        sys.modules["mmcv.ops"] = MagicMock()
        sys.modules["mmcv.ops.multi_scale_deform_attn"] = MagicMock()
        sys.modules["mmcv.ops.active_rotated_filter"] = MagicMock()
except ImportError:
    pass

# 注册自定义 Transform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class LoadImageFromNDArray:
    """从 numpy 数组加载图像的自定义 Transform"""
    
    def __init__(self, **kwargs):
        pass

    def __call__(self, results):
        img = results.get('img')
        if img is not None:
            if isinstance(img, list):
                img = img[0]
            h, w = img.shape[:2]
            results['img_shape'] = (h, w)
            results['ori_shape'] = (h, w)
        return results


from mmpose.apis import init_model
from mmcv.transforms import Compose


class PoseEstimator:
    """姿态估计器，使用 RTMPose (MMPose) 进行人体关键点识别"""
    
    # Halpe-26 数据集格式（26个关键点）
    # 默认配置内容
    DEFAULT_CONFIG_CONTENT = """
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
"""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cuda:0"
    ):
        """
        初始化姿态估计器
        
        Args:
            config_path: 配置文件路径。如果为 None，将使用默认配置并创建临时配置文件
            checkpoint_path: 模型权重文件路径。如果为 None，默认使用 weights/rtmpose_m_halpe26.pth
            device: 推理设备，'cuda:0' 或 'cpu'
        """
        repo_root = Path(__file__).resolve().parents[3]
        
        # 处理配置文件
        if config_path is None:
            config_dir = repo_root / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "rtmpose_m_halpe26.py"
            
            # 如果配置文件不存在，创建默认配置
            if not config_path.exists():
                with open(config_path, 'w', encoding='utf-8') as f:
                    f.write(self.DEFAULT_CONFIG_CONTENT)
                print(f"已创建默认配置文件: {config_path}")
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 处理模型权重路径
        if checkpoint_path is None:
            checkpoint_path = repo_root / "weights" / "rtmpose_m_halpe26.pth"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"模型权重文件不存在: {checkpoint_path}\n"
                f"请下载 RTMPose 模型权重文件并放置在 {checkpoint_path}"
            )
        
        # 确保权重目录存在
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载模型
        try:
            print(f"正在加载 RTMPose 模型...")
            print(f"  配置文件: {config_path}")
            print(f"  权重文件: {checkpoint_path}")
            print(f"  设备: {device}")
            
            self.model = init_model(str(config_path), str(checkpoint_path), device=device)
            self.test_pipeline = Compose(self.model.cfg.test_pipeline)
            self.device = device
            
            print(f"RTMPose 模型加载成功")
        except Exception as e:
            raise RuntimeError(f"无法加载 RTMPose 模型: {e}")
    
    def estimate(
        self,
        frame: np.ndarray,
        bboxes: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        对输入帧和检测框进行姿态估计
        
        Args:
            frame: 输入图像帧，形状为 (H, W, C) 的 numpy 数组，BGR 格式
            bboxes: 检测框数组，形状为 (N, 4) 或 (4,)，格式为 [x1, y1, x2, y2]
                   如果为 None 或空，返回 (None, None)
        
        Returns:
            Tuple[keypoints, scores]:
            - keypoints: 关键点坐标数组，形状为 (N, 26, 2)，如果检测失败返回 None
            - scores: 关键点置信度数组，形状为 (N, 26)，如果检测失败返回 None
                     如果输入 bboxes 为空，返回 (None, None)
        """
        # 容错处理：如果输入为空，返回空结果（参考旧代码风格）
        if frame is None or frame.size == 0:
            return None, None
        
        # 处理 bboxes 输入格式（容错处理）
        if bboxes is None:
            return None, None
        
        bboxes = np.array(bboxes)
        
        # 如果是一维数组，转换为二维
        if bboxes.ndim == 1:
            if bboxes.size < 4:
                return None, None
            bboxes = bboxes.reshape(1, -1)
        
        # 检查 bboxes 是否为空
        if bboxes.size == 0 or bboxes.shape[0] == 0:
            return None, None
        
        # 确保 bboxes 格式正确（至少要有 4 列：x1, y1, x2, y2）
        if bboxes.shape[1] < 4:
            return None, None
        
        # 选择面积最大的框
        if bboxes.shape[0] > 1:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            max_idx = np.argmax(areas)
            target_bbox = bboxes[max_idx:max_idx + 1]
        else:
            target_bbox = bboxes
        
        try:
            # 构建数据字典
            data_info = {
                'img': frame,
                'bbox': target_bbox,
                'bbox_score': np.ones(target_bbox.shape[0], dtype=np.float32),
                'category_id': 0,
            }
            
            # 数据预处理
            data = self.test_pipeline(data_info)
            
            # 准备输入
            inputs = data['inputs'].unsqueeze(0).to(self.device)
            data_samples = [data['data_samples']]
            
            # 模型推理（使用 val_step，与参考代码一致）
            with torch.no_grad():
                pose_results = self.model.val_step({
                    'inputs': inputs,
                    'data_samples': data_samples
                })
            
            # 提取结果
            if pose_results and len(pose_results) > 0:
                pred_sample = pose_results[0]
                
                # 检查是否有预测结果
                if hasattr(pred_sample, 'pred_instances') and len(pred_sample.pred_instances) > 0:
                    keypoints = pred_sample.pred_instances.keypoints[0]  # 形状: (26, 2)
                    scores = pred_sample.pred_instances.keypoint_scores[0]  # 形状: (26,)
                    
                    # 转换为 numpy 数组
                    if isinstance(keypoints, torch.Tensor):
                        keypoints = keypoints.cpu().numpy()
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().numpy()
                    
                    # 返回形状为 (1, 26, 2) 和 (1, 26) 的结果，保持与多框情况一致
                    return keypoints[np.newaxis, :], scores[np.newaxis, :]
            
            # 如果没有检测到关键点，返回 None（容错处理）
            return None, None
            
        except Exception as e:
            # 异常处理：返回空结果而不是抛出异常（参考旧代码的容错逻辑）
            print(f"警告: 姿态估计失败: {e}")
            return None, None
