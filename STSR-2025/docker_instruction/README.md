# Docker Submission Guide for STSR 2025

This guide provides step-by-step instructions for packaging your inference code and model into a Docker container for submission. Following these instructions is crucial for ensuring the reproducibility and successful evaluation of your algorithm.

[Chinese Version](./README_zh-CN.md) |

### **Overview of the Process**

You will need to provide a complete Docker environment that can take a 3D CBCT scan (`.nii.gz` file) as input and produce a segmentation mask as output. The process involves:
1.  **Preparing your files**: Organizing your code, model weights, and dependencies.
2.  **Configuring the Docker environment**: Modifying the provided template files.
3.  **Building the Docker image**: Using the `build.sh` script.
4.  **Testing the Docker image locally**: (Highly Recommended)
5.  **Exporting the Docker image**: Using the `export.sh` script for submission.

### **File Structure**

We provide a template folder to help you get started. Your final submission structure should look like this:

```
your_team_name/
├── Dockerfile              # (Provided, usually no need to change)
├── build.sh                # (Provided, set your team name)
├── export.sh               # (Provided, set your team name)
├── predict.sh              # (Provided, usually no need to change)
├── requirements.txt        # <-- ACTION REQUIRED: Add your dependencies
├── run_inference.py        # <-- ACTION REQUIRED: Implement your inference logic
├── your_model_weights.pth  # <-- ACTION REQUIRED: Add your trained model file(s)
└── model/                  # <-- ACTION REQUIRED: Add your model definition scripts
    └── UNet.py
```

### **Step-by-Step Instructions**

#### **Step 1: Set Your Team Name**

Open `build.sh` and `export.sh` and change the `TEAM_NAME` variable from `"teamname"` to your actual team name. This name will be used to tag your Docker image and name the final submission file.

```bash
# In build.sh and export.sh
TEAM_NAME="teamname" # Replace with your team name
```

#### **Step 2: Add Dependencies in `requirements.txt`**

List all Python packages required by your `run_inference.py` script in the `requirements.txt` file. The base Docker image already includes PyTorch and NumPy.

**For better reproducibility, we strongly recommend pinning the versions**, e.g., `monai==1.3.0`.

#### **Step 3: Place Your Model Files**

1.  **Model Definition**: Place the Python script(s) that define your model architecture (e.g., `UNet.py`) inside the `model/` directory.
2.  **Model Weights**: Place your trained model weights file (e.g., `your_model_weights.pth`) in the root directory.

#### **Step 4: Implement Inference Logic in `run_inference.py`**

This is the most critical step. Open `run_inference.py` and modify it according to your needs. The template is designed to be flexible.

1.  **Import Your Model**: Change the `from model.UNet import UNet` line to import your specific model class.
2.  **Specify Model Path**: Update the `MODEL_PATH` variable to match the filename of your model weights.
3.  **Implement `preprocess()`**: Add your image pre-processing logic (e.g., normalization, resizing) inside this function.
4.  **Implement `postprocess()`**: Add your prediction post-processing logic (e.g., resizing back to original shape, applying thresholding) inside this function.
5.  **Initialize Your Model**: Ensure the model initialization (`UNet(...)`) matches the architecture and parameters of your trained model.

#### **Step 5: Build Your Docker Image**

Once all files are configured, build the Docker image by running the `build.sh` script from your terminal:

```bash
sh build.sh
```

This process may take some time as it downloads the base image and installs all dependencies. Watch for any errors during the build process.

#### **Step 6: (Recommended) Test Your Docker Image Locally**

Before submitting, you should test your container to ensure it runs correctly.

1.  Create two folders on your local machine: `test_input` and `test_output`.
2.  Place a sample `.nii.gz` test file into the `test_input` folder.
3.  Run the following command, replacing `my-awesome-team` with your team name and adjusting the absolute paths to your test folders:

    ```bash
    docker container run \
      --gpus all \
      --name my-awesome-team-test \
      --rm \
      -v "/path/to/your/test_input":/inputs \
      -v "/path/to/your/test_output":/outputs \
      my-awesome-team:latest
    ```
4.  Check the `test_output` folder. A segmentation mask should have been generated. Also, check the command's log output for any errors.

#### **Step 7: Export for Submission**

If the local test is successful, run the `export.sh` script to create the final submission file:

```bash
sh export.sh
```

This will create a file named `your_team_name.tar.gz`. This is the file you need to submit to us.

### **Evaluation Environment**

Your submitted Docker container will be evaluated with the following commands. Note that you have only one chance to fix bugs if the container fails.

```bash
# 1. Load the submitted Docker image
docker load -i your_team_name.tar.gz

# 2. Run the container for inference
docker container run \
  --gpus all \
  -m 8G \
  --name your_team_name \
  --rm \
  -v $PWD/test_case_data/:/inputs \
  -v $PWD/your_team_name_outputs/:/outputs \
  your_team_name:latest
```
*Note: The actual evaluation command might vary slightly, but the core logic of mounting `/inputs` and `/outputs` will be the same.*