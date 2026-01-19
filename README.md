# VLN-UAV 2.5D/BEV Pipeline (AutoDL)

该仓库提供一个可复现的端到端脚本，用于从 **图像 + 深度 + 位姿** 构建 2.5D/BEV 地图（occupancy、height_max、semantic），并提供 A* 导航接口 Demo。

## 目录
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [端到端执行](#端到端执行)
- [输出说明](#输出说明)
- [导航接口 Demo](#导航接口-demo)
- [常见问题](#常见问题)

## 环境安装
```bash
cd /workspace/VLN-UAV
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据准备
### A) EuRoC MAV
示例根目录：`/root/autodl-tmp/sam-3d-objects/inputs/V2_01_medium/mav0`

需要包含：
- `cam0/data/*.png`, `cam1/data/*.png`
- `cam0/sensor.yaml`, `cam1/sensor.yaml`
- `state_groundtruth_estimate0/data.csv`

### B) TartanAir 子集
目录结构示例：
```
<root>/image_left/
<root>/depth_left/
<root>/seg_left/ (可选)
<root>/pose_left.txt
```

## 端到端执行
### EuRoC（双目 + 位姿）
```bash
python scripts/run_pipeline.py \
  --dataset euroc \
  --data-root /root/autodl-tmp/sam-3d-objects/inputs/V2_01_medium/mav0 \
  --output outputs/euroc_demo \
  --max-frames 200
```

### TartanAir（RGB + Depth + Pose + Seg）
```bash
python scripts/run_pipeline.py \
  --dataset tartanair \
  --data-root /path/to/tartanair/scene \
  --output outputs/tartan_demo \
  --max-frames 200 \
  --things-ids 1,2,3 \
  --stuff-ids 4,5,6
```

> `--things-ids`/`--stuff-ids` 用于将分割 label 投影到 BEV，并区分 things/stuff。若不提供则只输出语义 ID 栅格。

## 输出说明
输出目录包含：
- `bev_occupancy.npy/png`：占据栅格
- `bev_height_max.npy/png`：最大高度
- `bev_semantic.npy/png`：语义 ID 栅格（若输入有 seg）
- `bev_things.png` / `bev_stuff.png`：things/stuff 掩膜（若指定 ids）
- `meta.json`：运行参数、帧数与内参记录

## 导航接口 Demo
```bash
python scripts/run_navigation_demo.py \
  --map-dir outputs/euroc_demo \
  --start 50,50 \
  --goal 150,150 \
  --output outputs/euroc_demo/nav_path.png
```

## 常见问题
- **EuRoC 帧数不一致**：脚本会在 cam0/cam1 取交集，并用最近姿态对齐。
- **单帧效果呈扇形**：这是正常现象，多帧融合后会形成全局 BEV。
- **数据路径变动**：使用 `--data-root` 传入即可，无需改代码。
