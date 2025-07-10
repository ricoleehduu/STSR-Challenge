[View this document in Chinese (中文)](./README_zh-CN.md)

### **Local Evaluation Guide**

We provide a local evaluation tool to help you check your model's performance before submission.

**1. Prerequisites**

First, please ensure you have the required Python packages installed:

```bash
pip install numpy simpleitk scipy
```

You will need to place the compute_metrics.py and SurfaceDice.py scripts in the same directory.

**2. How to Run**

Run the evaluation script using the following command. Please replace the placeholders with your actual paths:

```
python compute_metrics.py -p <path/to/your/predictions> -g <path/to/ground_truth> -o <path/to/output_scores.json>
```

**Argument Descriptions:**

- -p, --prediction_dir: Path to the directory containing your generated prediction masks in NIfTI format.
- -g, --groundtruth_dir: Path to the directory containing the provided ground truth masks in NIfTI format.
- -o, --output_file: (Optional) Specify the filename for the output scores JSON file. Defaults to scores.json.

**Important Note**: Your prediction filenames must match the corresponding filenames in the ground truth folder exactly.

**Example:**
For example, if your predictions are in ./my_predictions and the provided test labels are in ./test_cases/ground_truth, you can run the script as follows:

```
python compute_metrics.py -p ./my_predictions -g ./test_cases/ground_truth
```

After the evaluation is complete, a file named scores.json will be generated in the current directory, containing the detailed scores for all metrics.
