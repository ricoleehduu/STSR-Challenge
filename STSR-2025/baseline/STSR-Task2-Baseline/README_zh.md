# 3D牙科配准：CBCT与口内扫描对齐

[README_en.md](./README.md)

本项目提供了一个端到端的深度学习流程，用于自动配准3D牙科模型，特别是将口内扫描（IOS，STL格式）与锥形束CT（CBCT，.nii.gz格式）进行对齐。项目的核心是一个基于PointNet++的双分支网络，它学习预测用于对齐的4x4刚性变换矩阵。

## 功能特性

- **端到端流程**: 涵盖从数据预处理、模型训练、推理到提交结果打包的全过程。
- **基于深度学习**: 利用双分支PointNet++架构学习稳健的几何特征以实现配准。
- **可配置化**: 所有的路径、超参数和实验设置都可以通过YAML配置文件轻松管理。
- **多中心数据处理**: 包含用于修正多中心数据集中常见坐标系不一致问题的脚本。
- **批量推理与打包**: 能够自动处理整个数据集（如验证集），并将结果打包成符合竞赛要求的`.zip`文件。
- **TensorBoard集成**: 可视化监控训练和验证过程中的损失曲线，更好地洞察模型性能。

## 项目结构

```
.
├── configs/                  # YAML 配置文件
│   ├── train_config.yaml
│   └── inference_config.yaml
├── data/                     # 数据加载与预处理
│   └── dataset.py
├── models/                   # 模型结构定义
│   ├── main_model.py
│   ├── pointnet2.py
│   └── registration_head.py
├── losses/                   # 损失函数定义
│   └── registration_loss.py
├── main_train.py             # 训练主脚本
├── main_inference_batch.py   # 批量推理与打包主脚本
├── requirements.txt          # Python 依赖
└── README_zh.md              # 本文档
```

## 环境设置与安装

### 1. 克隆仓库

```bash
git clone <你的仓库地址>
cd <仓库文件夹>
```

### 2. 创建虚拟环境 (推荐)

```bash
# 使用 Conda
conda create -n dental_reg python=3.9
conda activate dental_reg

# 或使用 venv
python -m venv venv
# Windows: venv\Scripts\activate | Linux/MacOS: source venv/bin/activate
```

### 3. 安装依赖

本项目依赖于PyTorch和PyTorch Geometric。请根据你的CUDA版本进行安装。

**第一步: 安装 PyTorch**
访问 [PyTorch官网](https://pytorch.org/get-started/locally/)，获取适合你环境的安装命令。例如，针对CUDA 11.8：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**第二步: 安装 PyTorch Geometric 依赖**
访问 [PyG安装指南](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) 获取安装说明。命令通常如下所示（请根据你的torch版本调整）：
```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
```

**第三步: 安装其余依赖**
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 数据准备

本项目假定你的数据已按特定结构组织，并已通过预处理脚本修正了坐标系不一致问题。最终的数据结构应如下：
```
<数据集根目录>/ (例如, Train-Labeled-Corrected)
├── Images/
│   ├── 001/
│   │   ├── CBCT.nii.gz
│   │   ├── lower.stl
│   │   └── upper.stl
│   └── ...
└── Labels/
    ├── 001/
    │   ├── lower_gt.npy
    │   └── upper_gt.npy
    └── ...
```
`_gt.npy` 文件包含4x4的真值变换矩阵。

### 2. 参数配置

修改 `configs/` 目录下的配置文件：

- **`configs/train_config.yaml`**:
    - 设置 `train_data_root` 和 `val_data_root` 指向你的训练和验证数据路径。
    - 定义一个 `experiment_name`。所有输出（权重、日志）都将保存在 `output_dir/experiment_name` 下。
    - 选择要训练的 `jaw_type`（'lower' 或 'upper'）。

- **`configs/inference_config.yaml`**:
    - 设置 `inference_data_root` 指向你希望进行推理的数据集。
    - 指定 `lower_jaw_experiment_name` 和 `upper_jaw_experiment_name` 以加载正确的已训练模型。

### 3. 模型训练

你需要为上颌和下颌分别训练独立的模型。

**训练下颌模型:**
1. 在 `train_config.yaml` 中，设置 `jaw_type: "lower"` 和 `experiment_name: "lower_jaw_v1"`。
2. 运行训练脚本:
   ```bash
   python main_train.py
   ```
3. 使用TensorBoard监控训练过程:
   ```bash
   tensorboard --logdir=./experiments
   ```

**训练上颌模型:**
1. 在 `train_config.yaml` 中，修改 `jaw_type: "upper"` 和 `experiment_name: "upper_jaw_v1"`。
2. 再次运行训练脚本。

### 4. 批量推理与打包

当两个模型都训练好后，你可以为整个数据集（如验证集）生成预测结果，并创建提交所需的zip文件。

1. 正确配置 `configs/inference_config.yaml`。
2. 运行批量推理脚本:
   ```bash
   python main_inference_batch.py
   ```
3. 一个名为 `prediction.zip` 的文件将在你的项目根目录下生成，其内部结构符合竞赛提交要求。

## 未来优化方向

本项目提供了一个强大的基线模型。以下是一些潜在的改进方向：

1.  **统一模型**: 训练一个单一模型来同时处理上颌和下颌，而不是两个独立模型。这可以通过向网络输入一个“牙颌类型”的嵌入向量来实现。
2.  **更先进的特征提取器**: 探索更先进的点云网络主干，如KP-Conv或基于Transformer的架构（例如Point-BERT），它们可能能捕捉到更复杂的几何关系。
3.  **端到端精调**: 在深度学习模型预测出初始位姿后，集成一个传统的ICP（迭代最近点）算法或一个可学习的精调模块，以进一步提升配准精度。
4.  **更复杂的损失函数**: 尝试在变换后的点云上使用其他损失函数，如倒角距离（Chamfer Distance）或推土机距离（EMD）。同时，可以考虑增加一个特征相似度损失来指导编码器的学习。
5.  **处理部分扫描数据**: 当前模型假设扫描数据相对完整。对于高度残缺的数据，专注于稳健特征匹配的技术（如3DMatch或D3Feat）可能会更有效。
6.  **无监督/自监督学习**: 如果有标签数据稀缺，可以探索在大量无标签数据上进行自监督预训练，以学习强大的几何表征，然后再在有标签的集合上进行微调。

---