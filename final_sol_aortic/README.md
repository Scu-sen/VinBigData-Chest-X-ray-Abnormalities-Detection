# Aortic Specialist: Steps to Reproduce

We describe the steps to reproduce the predictions for aortic enlargement class from our specialized detector. Along the way, we also describe the steps to generate the class 10 (Pleural Effusion) predictions that we ended up using in our final solution.

Optionally, we provide steps and scripts to preprocess data and train the detectors.

## System Overview

Training and inference were performed on a system with the following specifications and setup:
- Ubuntu 20.04.1, 64-bit
- Python 3.7.9
- CUDA 10.1
- cuDNN 7.6.5
- Pytorch 1.7.0
- Hardware: 1xGTX 1080Ti, 32 GB RAM

Required packages are listed in the `requirements.txt` file.

## Step 1: Data Preparation

### a. Data Download
First, download the [competition data](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data) and place the files in the `data/input/` directory.

### b. Convert DICOM to PNG
Next, convert the DICOM from both train and test sets into PNG using the provided `dicom_to_png.py` script. You will need to generate PNG images of both 512 and 1024 square dimensions. You can do so with the following commands
```sh
python dicom_to_png.py --img-path data/input/train --img-size 512 --output-file train512.zip
python dicom_to_png.py --img-path data/input/test --img-size 512 --output-file test512.zip
python dicom_to_png.py --img-path data/input/train --img-size 1024 --output-file train1024.zip
python dicom_to_png.py --img-path data/input/test --img-size 1024 --output-file test1024.zip
```
Extract the generated zipfiles into the following directories accordingly:
```
data/extracted_images/512/train/
data/extracted_images/512/test/
data/extracted_images/1024/train/
data/extracted_images/1024/test/
```
### c. Obtain Original Dimensions of CXR Images (Optional)
For each image, we wish to extract the original dimensions. The files for this step (`{train, test}_meta.csv`) are already provided in `data/input/`. However, if you wish to generate them yourself, you can run the following commands:
```
python extract_image_dims.py --split train --output-file data/input/train_meta.csv
python extract_image_dims.py --split test --output-file data/input/test_meta.csv
```


## Step 2: Detector Training (Optional)

Final trained weights for our detectors are provided in `data/yolo/weights/`. We also provide the YOLOv5 dataset yaml files used when training these weights in `data/yolo/yolo_configs/`.

If you wish to train these models yourself, please follow the steps below. In our final solution, we have 7 separate models from this step: 2 "full" detectors train on R8 annotations (input image size 512 and 1024, respectively), and 5 "specialist" detectors trained on non-R8/R9/R10 annotations (5 folds).

### a. Preprocess Non-R8/R9/R10 Annotations (Optional) 
For our specialized detector, we combined annotations from non-R8/R9/R10 radiologists using weighted boxes fusion, for training. This combined annotations are available in `data/input/nonR8R9R10_wbf.csv`. If you wish to generate these yourself, you can run the following:
```
python box_ensembling_preprocess.py --output-file data/input/nonR8R9R10_wbf.csv
```

### b. Prepare Data for YOLOv5 Training 
Run the following commands to prepare training data for the full R8 detectors (input 512 and 1024):
```
python data_prep_for_detector_R8.py
```
and the following to prepare training data for the specialized detectors (5 folds):
```
python data_prep_for_detector_specialist.py
```


### c. Train Detectors
In the `yolov5` directory, running the following commands mimics the training configuration that we used for the full R8 detectors:
```
> python train.py --img 512 --batch 16 --epochs 30 --data ../data/yolo/yolo_configs/vbd_R8_512.yaml --weights ../data/yolo/weights/yolov5x.pt
> python train.py --img 1024 --batch 4 --epochs 60 --data ../data/yolo/yolo_configs/vbd_R8_1024.yaml --weights ../data/yolo/weights/yolov5x.pt
```
and for the specialized detectors:
```
> python train.py --img 512 --batch 16 --epochs 100 --data ../data/yolo/yolo_config/svbd_aortic_fold{0,1,2,3,4}.yaml --weights ../data/yolo/weights/best_R8_512.yaml
```

These steps will reproduce the provided weights found in `data/yolo/weights/`.



## Step 3: Inference

Run the following to perform inference + ensembling on the specialized detectors for aortic enlargement.
```
python inference_aortic.py
```
The resulting csv file `aortic_final_fixed.csv` will be used in our final post-processing (PP) step.

As part of our final PP, we also used class 10 (Pleural Effusion) predictions from our full R8 detectors. 
You can generate the file (`cls10_final.csv`) by running the following command.
```
python inference_R8.py
```



