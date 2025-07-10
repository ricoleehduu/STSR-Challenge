[English Version](./README.md) 

# STSR 2025 Docker 提交指南

本指南将为您提供详细的步骤说明，指导您如何将您的推理代码和模型打包成一个用于提交的 Docker 容器。遵循这些说明对于确保您算法的可复现性和成功评估至关重要。

### **提交流程概览**

您需要提供一个完整的 Docker 环境，该环境能够接收一个 3D CBCT 扫描文件（`.nii.gz` 格式）作为输入，并生成对应的分割掩码作为输出。整个过程包括：
1.  **准备文件**：组织您的代码、模型权重和依赖项。
2.  **配置 Docker 环境**：修改我们提供的模板文件。
3.  **构建 Docker 镜像**：使用 `build.sh` 脚本。
4.  **本地测试 Docker 镜像**：（强烈推荐）
5.  **导出 Docker 镜像**：使用 `export.sh` 脚本生成最终提交文件。

### **文件结构**

我们提供了一个模板文件夹来帮助您开始。您最终的项目结构应如下所示：

```
your_team_name/
├── Dockerfile              # (已提供，通常无需修改)
├── build.sh                # (已提供，需设置您的队名)
├── export.sh               # (已提供，需设置您的队名)
├── predict.sh              # (已提供，通常无需修改)
├── requirements.txt        # <-- 需要您操作：添加您的依赖项
├── run_inference.py        # <-- 需要您操作：实现您的推理逻辑
├── your_model_weights.pth  # <-- 需要您操作：添加您训练好的模型文件
└── model/                  # <-- 需要您操作：添加您的模型定义脚本
    └── UNet.py
```

### **分步操作指南**

#### **第一步：设置您的队名**

打开 `build.sh` 和 `export.sh` 文件，将 `TEAM_NAME` 变量的值从 `"teamname"` 修改为您的实际队名。这个名称将用于标记您的 Docker 镜像和命名最终的提交文件。

```bash
# 在 build.sh 和 export.sh 文件中
TEAM_NAME="teamname" # 替换为您的队名
```

#### **第二步：在 `requirements.txt` 中添加依赖项**

请在 `requirements.txt` 文件中列出您的 `run_inference.py` 脚本所需的所有 Python 包。基础的 Docker 镜像已经包含了 PyTorch 和 NumPy。

**为了更好的可复现性，我们强烈建议您锁定包的版本号**，例如 `monai==1.3.0`。

#### **第三步：放置您的模型文件**

1.  **模型定义**：将定义您模型架构的 Python 脚本（例如 `UNet.py`）放入 `model/` 目录中。
2.  **模型权重**：将您训练好的模型权重文件（例如 `your_model_weights.pth`）放入项目的根目录。

#### **第四步：在 `run_inference.py` 中实现推理逻辑**

这是最关键的一步。请打开 `run_inference.py` 并根据您的需求进行修改。我们提供的模板设计得非常灵活。

1.  **导入您的模型**：修改 `from model.UNet import UNet` 这一行，以导入您自己的模型类。
2.  **指定模型路径**：更新 `MODEL_PATH` 变量，使其与您的模型权重文件名匹配。
3.  **实现 `preprocess()` 函数**：在此函数中添加您的图像预处理逻辑（例如，归一化、尺寸调整等）。
4.  **实现 `postprocess()` 函数**：在此函数中添加您的预测后处理逻辑（例如，将尺寸调整回原始大小、应用阈值等）。
5.  **初始化您的模型**：确保模型初始化代码（例如 `UNet(...)`）与您训练好的模型的架构和参数相匹配。

#### **第五步：构建您的 Docker 镜像**

当所有文件都配置好后，在您的终端中运行 `build.sh` 脚本来构建 Docker 镜像：

```bash
sh build.sh
```

这个过程可能需要一些时间，因为它需要下载基础镜像并安装所有依赖项。请留意构建过程中的任何错误信息。

#### **第六步：（推荐）在本地测试您的 Docker 镜像**

在提交之前，您应该在本地测试您的容器以确保其能正常运行。

1.  在您的本地机器上创建两个文件夹：`test_input` 和 `test_output`。
2.  将一个示例的 `.nii.gz` 测试文件放入 `test_input` 文件夹。
3.  运行以下命令，请将 `my-awesome-team` 替换为您的队名，并将路径修改为您本地测试文件夹的绝对路径：

    ```bash
    docker container run \
      --gpus all \
      --name my-awesome-team-test \
      --rm \
      -v "/path/to/your/test_input":/inputs \
      -v "/path/to/your/test_output":/outputs \
      my-awesome-team:latest
    ```
4.  检查 `test_output` 文件夹。如果一切正常，里面应该会生成一个分割掩码文件。同时，检查命令行的日志输出以确认没有错误。

#### **第七步：导出以提交**

如果本地测试成功，运行 `export.sh` 脚本来创建最终的提交文件：

```bash
sh export.sh
```

这将创建一个名为 `您的队名.tar.gz` 的文件。这正是您需要提交给我们的文件。

### **评测环境**

您提交的 Docker 容器将通过以下命令进行评测。请注意，如果容器运行失败，您将只有一次修复 Bug 的机会。

```bash
# 1. 加载您提交的 Docker 镜像
docker load -i your_team_name.tar.gz

# 2. 运行容器进行推理
docker container run \
  --gpus all \
  -m 8G \
  --name your_team_name \
  --rm \
  -v $PWD/test_case_data/:/inputs \
  -v $PWD/your_team_name_outputs/:/outputs \
  your_team_name:latest
```
*注意：实际的评测命令可能略有不同，但挂载 `/inputs` 和 `/outputs` 目录的核心逻辑将保持不变。*