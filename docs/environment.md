# fytennis 运行环境记录

本项目在 Windows 的 Conda 环境 `fytennis` 中运行：

```bash
conda activate fytennis
python main.py
```

2026-09-08 检查到的直接依赖版本：

```text
Python environment: C:\Users\Administrator\miniconda3\envs\fytennis
fastapi==0.109.2
uvicorn==0.27.1
python-multipart==0.0.9
ultralytics==8.3.243
opencv-python==4.11.0.86
numpy==1.26.4
pandas==2.3.3
scipy==1.15.3
matplotlib==3.10.8
moviepy==2.2.1
torch==2.11.0.dev20260108+cu128
torchvision==0.25.0.dev20260108+cu128
mmcv==2.2.0
mmpose==1.3.2
mmengine==0.10.7
pydantic==2.6.1
```

PyTorch 和 torchvision 是 CUDA 12.8 开发版本，不能假设普通 `pip install torch`
会得到相同环境。迁移机器时应先按目标GPU安装匹配的PyTorch，再安装其余依赖。

后端只保存图片、不显示Matplotlib窗口。自动测试建议使用非交互式后端：

```powershell
$env:MPLBACKEND = "Agg"
python -m unittest discover -s tests -v
```
