This is the detection part for the [VinBigData Competition](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/overview).

# structure of the project
Here we assume that the project root is $detection_model, then the structure of project is as follows:
```
$detection_model
|--vinbigdata_configs
|--models
|--utils
|--requirements.txt
|--train.py
|--inference_dbtex.py
|--data
|--|--vinbiddata
|--|--|--images
|--|--|--|--train
|--|--|--|--test
|--|--|--annotations
|--|--|--|--vinbig_test.json
|--|--|--|--sample_submission.csv
|--|--|--|--new_Kfold_annotations
|--|--|--|--my_kfold_annotations
```

We have prepared the corresponding train/val annotations.


# initialize the environment
- 1.clone the mish-cuda repository  
```
cd $detection_model/../ (or anywhere you want)
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install
```

- 2.install the requirements
```
cd $detection_model
pip install requirements.txt
``` 

#DATA SETUP
We assume that the original structure of VinBigData dataset are as follows (the same as downloading from the
Kaggle competition [platform](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data]) 
```
$VinBigData_Dataset
|--train.csv
|--sample_submission.csv
|--train
|--test
```
Please run the following code, `preprocess_data.py` convert the X-ray images format from 
`.dicom` to `.png` for train and test set. We assume that dataset output directory is `$OUT_DIR`

```
python utils/vinbigdata/preprocess_data.py \
--src_dir $VinBigData_Dataset
--out_dir $OUT_DIR ( This parameter is optional)
``` 

The `$VinBigData_Dataset` is the path to [competition data](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data). Note that you can set `$OUT_DIR` to anywhere you want, or you can just leave it to be empty.
The default `$OUT_DIR` is `$detection_model/data/vinbigdata/images`
The structure of output directory is as follows:

```
$OUT_DIR
|--vinbig_train.json
|--vinbig_test.json
|--train
|--test
```

If you have set `$OUT_DIR` parameter, then please make a symlink to have the 
same data structure mentioned above, as follows:

```
cd $detection_model
mkdir data/vinbigdata
ln -s $OUT_DIR $detection_model/data/vinbigdata/images
```

#MODEL BUILD: There are two options to produce the results detection model.
- 1.ordinary prediction
    a) expect this to run for a few hours
    b) uses model checkpoints
- 2.retrain models
    a) expect this to run about a week
    b) trains all models from scratch
    c) follow this with (2) to produce entire solution from scratch

shell command to run each build is below
##1) ordinary prediction
- 1.Please run the following command to inference the detection results.

```
cd $detection_model
sed -i 's/\r//g' vinbigdata_configs/inferencel_model.sh
sh vinbigdata_configs/inferencel_model.sh
```

- 2.After running the above command, the output csv files of 11 models will be stored in 
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


##2) retrain models
```
cd $detection_model
sed -i 's/\r//g' vinbigdata_configs/train_models.sh
sh vinbigdata_configs/train_models.sh
```
Note that this may take a long time to run. The trained checkpoints will be saved under the
`$detection_model/runs` directory.


##3) inference the retrained models
- 1.Run the followng command:
```
cd $detection_model
sed -i 's/\r//g' vinbigdata_configs/new_train_inference.sh
sh vinbigdata_configs/new_train_inference.sh
```

- 2.After running this command, the inference results will be saved into 
`$detection_model/inference/NewTrained_results`. And also, you need merge the results.

```
cd $detection_model
python utils/vinbigdata/merge_csv_results.py \
--src_dir $detection_model/inference/NewTrained_results
--out_dir $OUT_DIR (this is optional)
```

`$OUT_DIR` is the path to save merging results of the 11 csv files. The output will be saved into
 `vinbigdata_classifierPP_2021/detector_preds` by default, with the name of `final_before_pp.csv`.
