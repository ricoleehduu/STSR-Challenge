# 3D Dental Registration: CBCT to Intraoral Scans

[README_zh](./README_zh.md)

This project provides an end-to-end deep learning pipeline for automatically registering 3D dental models, specifically aligning Intraoral Scans (IOS, in STL format) with Cone-Beam Computed Tomography (CBCT, in .nii.gz format). The core of the project is a PointNet++ based dual-branch network that learns to predict the 4x4 rigid transformation matrix.

## Features

- **End-to-End Pipeline**: From data preprocessing to training, inference, and submission packaging.
- **Deep Learning-Based**: Utilizes a dual-branch PointNet++ architecture to learn robust geometric features for registration.
- **Configurable**: All paths, hyperparameters, and experiment settings can be easily managed via YAML configuration files.
- **Multi-Center Data Handling**: Includes scripts to correct coordinate system inconsistencies often found in multi-center datasets.
- **Batch Inference & Packaging**: Automatically processes an entire dataset (e.g., validation set) and packages the results into a competition-ready `.zip` file.
- **TensorBoard Integration**: Monitors training and validation loss curves for better insight into the model's performance.

## Project Structure

```
.
├── configs/                  # YAML configuration files
│   ├── train_config.yaml
│   └── inference_config.yaml
├── data/                     # Data loading and preprocessing
│   └── dataset.py
├── models/                   # Model architectures
│   ├── main_model.py
│   ├── pointnet2.py
│   └── registration_head.py
├── losses/                   # Loss function definitions
│   └── registration_loss.py
├── main_train.py             # Main script for training
├── main_inference_batch.py   # Main script for batch inference and packaging
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup and Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Using Conda
conda create -n dental_reg python=3.9
conda activate dental_reg

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

The project relies on PyTorch and PyTorch Geometric. Please install them according to your CUDA version.

**Step 1: Install PyTorch**
Visit the [official PyTorch website](https://pytorch.org/get-started/locally/) and get the command for your specific setup. For example, for CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 2: Install PyTorch Geometric Dependencies**
Visit the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for instructions. The command will look similar to this:
```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html # Adjust to your torch version
```

**Step 3: Install Remaining Requirements**
```bash
pip install -r requirements.txt
```

## How to Use

### 1. Data Preparation

This project assumes your data has a specific structure and has been pre-processed to correct for coordinate system inconsistencies. The final data structure should be:
```
<dataset_root>/ (e.g., Train-Labeled)
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
The `_gt.npy` files contain the 4x4 ground truth transformation matrices.

### 2. Configuration

Modify the configuration files in the `configs/` directory:

- **`configs/train_config.yaml`**:
    - Set `train_data_root` and `val_data_root` to your training and validation data paths.
    - Define an `experiment_name`. All outputs (weights, logs) will be saved under `output_dir/experiment_name`.
    - Choose the `jaw_type` ('lower' or 'upper') to train.

- **`configs/inference_config.yaml`**:
    - Set `inference_data_root` to the dataset you want to run inference on.
    - Specify the `lower_jaw_experiment_name` and `upper_jaw_experiment_name` to load the correct trained models.

### 3. Training

You need to train separate models for the upper and lower jaws.

**To train the lower jaw model:**
1. In `train_config.yaml`, set `jaw_type: "lower"` and `experiment_name: "lower_jaw_v1"`.
2. Run the training script:
   ```bash
   python main_train.py
   ```
3. Monitor the training progress with TensorBoard:
   ```bash
   tensorboard --logdir=./experiments
   ```

**To train the upper jaw model:**
1. In `train_config.yaml`, change `jaw_type: "upper"` and `experiment_name: "upper_jaw_v1"`.
2. Run the training script again.

### 4. Batch Inference and Packaging

Once both models are trained, you can generate predictions for a dataset (e.g., the validation set) and create a submission zip file.

1. Configure `configs/inference_config.yaml` correctly.
2. Run the batch inference script:
   ```bash
   python main_inference_batch.py
   ```
3. A `prediction.zip` file will be created in your project root, formatted according to competition requirements.

## Potential Improvements

This project provides a strong baseline. Here are some directions for future improvements:

1.  **Unified Model**: Instead of training two separate models, a single model could be trained to handle both jaws. This can be achieved by adding a "jaw type" embedding as an additional input to the network.
2.  **Advanced Feature Extractors**: Explore more advanced point cloud backbones like KP-Conv or Transformer-based architectures (e.g., Point-BERT) which might capture more complex geometric relationships.
3.  **End-to-End Refinement**: Integrate an ICP (Iterative Closest Point) or a learned refinement module after the initial deep learning-based prediction to further improve accuracy.
4.  **More Sophisticated Loss Functions**: Experiment with other loss functions, such as Chamfer distance or Earth Mover's Distance (EMD), on the transformed point clouds. Also, consider adding a loss on feature similarity to guide the encoders.
5.  **Handling Partial Scans**: The current model assumes relatively complete scans. For highly partial data, techniques focusing on robust feature matching (like 3DMatch or D3Feat) could be more effective.
6.  **Unsupervised/Self-Supervised Learning**: If labeled data is scarce, explore self-supervised pre-training on large amounts of unlabeled data to learn powerful geometric representations before fine-tuning on the labeled set.

