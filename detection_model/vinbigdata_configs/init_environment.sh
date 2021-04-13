#echo "installing mish-cuda"
#cd /mnt/group-ai-medical-2/private/zehuigong/torch_code
#git clone https://github.com/JunnYu/mish-cuda
#cd mish-cuda
#python setup.py build install

pip install pycocotools
pip install terminaltables
pip install matplotlib==3.0.1

#cd ScaledYOLOv4/data
#echo "preparing dataset for YOLOV4"
#mkdir vinbigdata
#mkdir vinbigdata/annotations
#mkdir vinbigdata/images
#
#root_dir=/mnt/group-ai-medical-2/private/zehuigong/dataset1/VinBigdata_AbnormDetect/processed_data
#echo "linking images..."
#ln -s ${root_dir}/train ./vinbigdata/images/
#ln -s ${root_dir}/test ./vinbigdata/images/
#
#echo "linking annotations..."
#ln -s ${root_dir}/Kfold_annotations ./vinbigdata/annotations/old_3fold
#ln -s ${root_dir}/new_Kfold_annotations ./vinbigdata/annotations/
#ln -s ${root_dir}/vinbig_test_cls.json ./vinbigdata/annotations/
#ln -s ${root_dir}/rad_annotations/vinbig_duplicateImgs.json ./vinbigdata/annotations/

#echo preparing SIIM-ACR dataset
#root_dir=/mnt/group-ai-medical-2/private/zehuigong/dataset1/SIIM_ACR
#mkdir data/siim_acr
#mkdir data/siim_acr/annotations
#mkdir data/siim_acr/images
#cd data/siim_acr
#ln -s ${root_dir}/dicom_images_train ./images/
#ln -s ${root_dir}/processed_data/annotations/SiimACR_annos.json ./annotations/

#echo preparing AAFMA dataset
#root_dir=/mnt/group-ai-medical-2/private/zehuigong/dataset1/A_AFMA_Detection
#mkdir data/aafma
#mkdir data/aafma/annotations
#mkdir data/aafma/images
#cd data/aafma
#ln -s ${root_dir}/train ./images/
#ln -s ${root_dir}/processed/fold_annotations ./annotations/

echo preparing RSNA dataset
root_dir=/mnt/group-ai-medical-2/private/zehuigong/dataset1/RSNA
mkdir data/rsna
#mkdir data/rsna/annotations
mkdir data/rsna/images
cd data/rsna
ln -s ${root_dir}/processed_data/train ./images/
ln -s ${root_dir}/processed_data/annotations ./annotations