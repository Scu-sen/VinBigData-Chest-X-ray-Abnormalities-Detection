kaggle VinBigData Chest X-ray Abnormalities Detection 3rd place code

Hello!

Below you can find a outline of how to reproduce my solution for the <VinBigData Chest X-ray Abnormalities Detection> competition.
If you run into any trouble with the setup/code or have any questions please contact me at <zehuigong@foxmail.com>

#ARCHIVE CONTENTS
- detection_model: this directory contains corresponding code and pre-trained weights to reproduce the results of
 11 detection models. 
- final_sol_aortic: contains corresponding code and pre-trained weights to reproduce the results of specialized detector
for aortic. 
- vinbigdata_classifierPP_2021: contains corresponding code and pre-trained weights to reproduce the results 
of multi-class classifier and the post-processing method.
- README.md: Details about how to reproduce the solution.

#HARDWARE: (The following specs were used to create the original solution)
Linux version: 4.14.105-1-tlinux3-0013
CPU Information: Intel(R) Xeon(R) Gold 6133 CPU @ 2.50GHz (cpu cores: 20, 359.58 GB memory)
GPU information: 8 x NVIDIA Tesla V100 (32 GB memory)

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.6.9
CUDA 10.1.243
cuddn 7.6.4
nvidia drivers: 440.64.00
GCC, G++ >= 4.9.4


#Environment Preparation
- 1. For the detection model, please refer `$ThirdPlaceSolutionCode/detection_model` for the details of preparing the environment.
- 2. For the specialized detector, Please refer `$ThirdPlaceSolutionCode/final_sol_aortic` for the details of preparing the environment.
- 3. For the multi-class classifier, Please refer `$ThirdPlaceSolutionCode/vinbigdata_classifierPP_2021` for the details of preparing the environment.

#MODEL BUILD: There are three options to produce the solution.
- 1.very fast prediction
    a) runs in a few minutes
    b) uses precomputed neural network predictions
- 2.ordinary prediction
    a) expect this to run for many hours
    b) uses binary model files
- 3.retrain models
    a) expect this to run about a few days
    b) trains all models from scratch
    c) follow this with (2) to produce entire solution from scratch


shell command to run each build is below

##1) very fast prediction
The corresponding code for fast prediction is under `./vinbigdata_classifierPP_2021/src/` directory.
<code> python postproc.py </code>
- Required input files: 
    - 3 \* 5 multilabel classifier prediction CSVs (1024, 1024_with_fixed_aspect_ratio, 1280_with_fixed_aspect_ratio) whose folder path specified by the `cls_dfs_path` variable.
    - A detector prediction CSV in submission format without any post-processing whose path specified by the `detector_preds_path` variable.
    - specialized predictions for class0 and class10 whose path specified by the `detector_preds_path` variable.
    - A list of image IDs determined based on DICOM metadata which very likely to be "normal" stated in `test_0.139000.csv`.
- Output: A CSV file in submission format saved in the folder specified by the `save_path` variable.

##2) ordinary prediction
There are three components for our solution to the VinBigData competition, i.e., detection model, 
specialized detector for cls 0 and 10, and the multi-class classifier and post-processing part.

- 1.Detection model.
    - 1.Please run the following command to inference the detection results.
    
    ```
    cd $detection_model
    sed -i 's/\r//g' vinbigdata_configs/inferencel_model.sh
    sh vinbigdata_configs/inferencel_model.sh
    ```
    
    - 2.$detection_model is the path to directory of detection_model. After running the above command, the output csv files of 11 models will be stored in 
    `$detection_model/inference/PreTrained_results`, then you need to merge the results into one csv.
    As follows:
    
    ```
    cd $detection_model
    python utils/vinbigdata/merge_csv_results.py \
    --src_dir $detection_model/inference/PreTrained_results
    --out_dir $OUT_DIR (this is optional)
    ```
    `$OUT_DIR` is the path to save merging results of the 11 csv files. The output will be saved into
    `vinbigdata_classifierPP_2021/detector_preds` by default, with the name of `final_before_pp.csv`.
    
- 2.Specialized detector.
    - 1.Run the following to perform inference + ensembling on the specialized detectors for aortic enlargement.
    ```
    cd $final_sol_aortic
    python inference_aortic.py
    ```
    The resulting csv file `aortic_final_fixed.csv` will be used in our final post-processing (PP) step.
    
    - 2.As part of our final PP, we also used class 10 (Pleural Effusion) predictions from our full R8 detectors. 
    You can generate the file (`cls10_final.csv`) by running the following command.
    ```
    cd $final_sol_aortic
    python inference_R8.py
    ```
- 3.Post-processing
    ```
    cd $vinbigdata_classifierPP_2021/src
    python multilabel_inference.py
    python postproc.py
    ```
    There are som paramaters you need to note about the above two python files.
    - 1.<code> python multilabel_inference.py </code> (open the file for details, the variable is at the top of the file)
        - Input1: All testing images converted to PNG format stored in a folder specified by the `image_path` variable.
        - Input2: 5 fold \* 15 models in .pth format saved in the folder specified by the `model_path` variable.
        - Input3: `test_meta.csv` from the competition dataset (or simply any csv file with all image_ids)
        - Output: 5 prediction CSVs (1 per fold) saved in the folder specified by the `save_path` variable.
    - 2. <code> python postproc.py </code> (open the file for details, the variable is at the top of the file)
        - Input1: 3 \* 5 multilabel classifier prediction CSVs (1024, 1024_with_fixed_aspect_ratio, 1280_with_fixed_aspect_ratio) whose folder path specified by the `cls_dfs_path` variable.
        - Input2: A detector prediction CSV in submission format without any post-processing whose path specified by the `detector_preds_path` variable.
        - Input3: specialized predictions for class0 and class10 whose path specified by the `detector_preds_path` variable.
        - Input4: A list of image IDs determined based on DICOM metadata which very likely to be "normal" stated in `test_0.139000.csv`.
        - Output: A CSV file in submission format saved in the folder specified by the `save_path` variable.


##3) retrain models
- 1.Detection model.
    -1. Run the following command.
    ```
    cd ThirdPlaceSolutionCode/detection_model
    sed -i 's/\r//g' vinbigdata_configs/train_models.sh
    sh vinbigdata_configs/train_models.sh
    ```
    Note that this may take a long time to run. The trained checkpoints will be saved under the
    `ThirdPlaceSolutionCode/detection_model/runs` directory.

- 2.Specialized detector.
    - 1.Firstly,
    ```
    cd ThirdPlaceSolutionCode/final_sol_aortic
    ```
    
    - 2.Running the following commands mimics the training configuration that we used for the full R8 detectors:
    ```
    > python train.py --img 512 --batch 16 --epochs 30 --data ../data/yolo/yolo_configs/vbd_R8_512.yaml --weights ../data/yolo/weights/yolov5x.pt
    > python train.py --img 1024 --batch 4 --epochs 60 --data ../data/yolo/yolo_configs/vbd_R8_1024.yaml --weights ../data/yolo/weights/yolov5x.pt
    ```
    - 3.for the specialized detectors:
    ```
    > python train.py --img 512 --batch 16 --epochs 100 --data ../data/yolo/yolo_config/svbd_aortic_fold{0,1,2,3,4}.yaml --weights ../data/yolo/weights/best_R8_512.yaml
    ```
    
    These steps will reproduce the provided weights found in `data/yolo/weights/`.
    
- 3.Post-processing
    The training file of multi-class classifier is under `ThirdPlaceSolutionCode/vinbigdata_classifierPP_2021/src/multilabel_train.py`
    Please run the following command:
    ```
    cd $ThirdPlaceSolutionCode\vinbigdata_classifierPP_2021\src
    python multilabel_train.py
    ```
    **Note that there are some parameters**
    - Input1: All training images converted to PNG format stored in a folder specified by the `image_path` variable.
    - Input2: `multilabel_cls_train.csv` generated by `multilabel_preproc.py` whose path specified by the `csv_path` variable.
    - Input3: `multilabel_pos_weight.npy` generated by `multilabel_preproc.py` whose path specified by the `pos_weight_path` variable.  
    - Output: 5 fold \* 15 models in .pth format saved in the folder specified by the `save_path` variable.

- 4. Please follow the step 2) to re-inference the results of new trained models. Please note that for `detection_model`,
the output of 11 inference csv files are under directory `ThirdPlaceSolutionCode/detection_model/inference/NewTrained_results`. 
Please refer to `ThirdPlaceSolutionCode\detection_model\README.md`, part 3 `##3) inference the retrained models` for details. # VinBigData-Chest-X-ray-Abnormalities-Detection
