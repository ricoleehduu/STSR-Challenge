

[English Version](./README.md) | 中文版本

### **本地评估指南**

我们提供了一个本地评估工具，让您可以在提交前检查您的模型性能。

**1. 准备工作**

首先，请确保您已安装所需的 Python 包：

```bash
pip install numpy simpleitk scipy
```

您需要将 compute_metrics.py 和 SurfaceDice.py 这两个脚本放在同一个目录下。

**2. 运行评估**

使用以下命令运行评估脚本。请将路径替换为您的实际路径：

```
python compute_metrics.py -p <您的预测结果文件夹> -g <真实标签文件夹> -o <输出分数文件路径>
```

**参数说明:**

- -p, --prediction_dir: 存放您生成的 NIfTI 格式预测掩码的文件夹。
- -g, --groundtruth_dir: 存放我们提供的 NIfTI 格式真实掩码的文件夹。
- -o, --output_file: （可选）指定保存分数的 .json 文件名。默认为 scores.json。

**重要提示**: 您的预测文件名必须与 groundtruth 文件夹中对应的文件名完全一致。

**示例:**
假设您的预测文件在 ./my_predictions，我们提供的测试标签在 ./test_cases/ground_truth，您可以这样运行：

```
python compute_metrics.py -p ./my_predictions -g ./test_cases/ground_truth
```

评估完成后，一个名为 scores.json 的文件将会生成在当前目录，其中包含了所有指标的详细分数。