## 3D CBCT Semi-supervised Segmentation task
For 3D CBCT Semi-supervised Segmentation task, we there are some rules to be followed when submitting prediction results for validation set.

To get a score for the validation set, competitors should submit a .zip file to the system, which contains 40 nii.gz files, each corresponding to the instance segmentation prediction result of an CBCT image in the validation set.
All nii.gz files for the predicted results must be named in the following format:
Validation_xxxx_Mask.nii.gz or STS25_Validation_xxxx_Mask.nii.gz, where xxxx stand for the id of the validation image. The id between submitted predicted nii.gz file and original validation image must be consistent.
❗NOTICE❗ The output prediction nii.gz files must directly packaged into a single zip file, no sub-folder should be included.

The submitted evaluation results is exactly the same form to the training set mask nii.gz file.

Prediction.zip
    │
    ├── STS25_Validation_0001_Mask.nii.gz
    ├── STS25_Validation_0002_Mask.nii.gz
    ....
    ├── STS25_Validation_0040_Mask.nii.gz


## CBCT-IOS Registration

