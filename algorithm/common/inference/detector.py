"""
检测器模块 - 封装 YOLOv8 用于人体和网球检测
"""
from pathlib import Path
from typing import Union, Optional, Any
import numpy as np
from ultralytics import YOLO


class TennisDetector:
    """网球场景检测器，使用 YOLOv8 进行人体和网球检测"""
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        *,
        confidence: float = 0.25,
        iou: float = 0.45,
    ):
        """
        初始化检测器
        
        Args:
            model_path: 模型文件路径。如果为 None，默认使用 weights/yolov8m.pt。
            confidence: YOLO 人体检测置信度阈值。
            iou: NMS IoU 阈值。
        """
        if model_path is None:
            # 默认模型路径
            repo_root = Path(__file__).resolve().parents[3]
            model_path = repo_root / "weights" / "yolov8m.pt"
        
        model_path = Path(model_path)
        self.confidence = float(confidence)
        self.iou = float(iou)
        
        # 如果模型文件不存在，YOLO 会自动下载
        # 但我们可以先检查路径是否存在，给出友好提示
        if not model_path.exists():
            # 如果目录不存在，创建目录
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"警告: 模型文件 {model_path} 不存在，将自动下载 YOLOv8 模型...")
        
        # 加载 YOLO 模型
        # YOLO 会自动处理模型下载和加载
        try:
            self.model = YOLO(str(model_path))
            print(f"检测器已加载: {model_path}")
        except Exception as e:
            raise RuntimeError(f"无法加载 YOLO 模型 {model_path}: {e}")
    
    def detect(self, frame: np.ndarray) -> Any:
        """
        对输入帧进行检测
        
        Args:
            frame: 输入图像帧，形状为 (H, W, C) 的 numpy 数组，BGR 格式
            
        Returns:
            YOLO 的检测结果对象 (ultralytics.engine.results.Results)
            结果包含以下属性：
            - boxes: 边界框信息（包含 xyxy, conf, cls 等）
            - names: 类别名称字典
            - orig_shape: 原始图像形状
            可以通过 result.boxes.xyxy 获取边界框坐标
            可以通过 result.boxes.conf 获取置信度
            可以通过 result.boxes.cls 获取类别ID
        """
        if frame is None or frame.size == 0:
            raise ValueError("输入帧为空")
        
        # 执行检测
        # YOLO 的 predict 方法会返回一个 Results 对象列表（单张图像时只有一个元素）
        results = self.model.predict(
            frame,
            verbose=False,  # 不输出详细信息
            conf=self.confidence,
            iou=self.iou,
        )
        
        # 单张图像时，返回第一个结果对象
        return results[0] if results else None
