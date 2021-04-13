pretrained='data/yolov4-p6.pt'
GPUS=8
BATCH_SIZE=32
img_size=1280
data_cfg=vinbigdata_configs/data/vinbigdata_speRadId.yaml
model_cfg=vinbigdata_configs/model/yolov4-p6_vinbig.yaml
hyp_cfg=vinbigdata_configs/data/hyp.finetune.yaml

rad_id=rad_id8
for FOLD in 1 2;do
  exp_name=yolov4_p6_${rad_id}F${FOLD}

  echo training with ${rad_id} fold ${FOLD}
  python -m torch.distributed.launch --nproc_per_node ${GPUS} train.py --batch-size ${BATCH_SIZE} --epochs 100 \
  --img-size ${img_size} ${img_size} --data ${data_cfg} --cfg ${model_cfg} --weights ${pretrained} --sync-bn \
  --device 0,1,2,3,4,5,6,7 --name ${exp_name} --hyp ${hyp_cfg} --test_freq 2 --save_freq 20 --rad_id ${rad_id} --fold ${FOLD}
done

for FOLD in 3 4 5;do
  exp_name=yolov4_p6_${rad_id}F${FOLD}

  echo training with ${rad_id} fold ${FOLD}
  python -m torch.distributed.launch --nproc_per_node ${GPUS} train.py --batch-size ${BATCH_SIZE} --epochs 100 \
  --img-size ${img_size} ${img_size} --data ${data_cfg} --cfg ${model_cfg} --weights ${pretrained} --sync-bn \
  --device 0,1,2,3,4,5,6,7 --name ${exp_name} --hyp ${hyp_cfg} --test_freq 2 --save_freq 30 --rad_id ${rad_id} --fold ${FOLD}
done

rad_id=rad_id9
for FOLD in 4;do
  exp_name=yolov4_p6_${rad_id}F${FOLD}

  echo training with ${rad_id} fold ${FOLD}
  python -m torch.distributed.launch --nproc_per_node ${GPUS} train.py --batch-size ${BATCH_SIZE} --epochs 100 \
  --img-size ${img_size} ${img_size} --data ${data_cfg} --cfg ${model_cfg} --weights ${pretrained} --sync-bn \
  --device 0,1,2,3,4,5,6,7 --name ${exp_name} --hyp ${hyp_cfg} --test_freq 2 --save_freq 20 --rad_id ${rad_id} --fold ${FOLD}
done

rad_id=rad_id10
for FOLD in 1;do
  exp_name=yolov4_p6_${rad_id}F${FOLD}

  echo training with ${rad_id} fold ${FOLD}
  python -m torch.distributed.launch --nproc_per_node ${GPUS} train.py --batch-size ${BATCH_SIZE} --epochs 100 \
  --img-size ${img_size} ${img_size} --data ${data_cfg} --cfg ${model_cfg} --weights ${pretrained} --sync-bn \
  --device 0,1,2,3,4,5,6,7 --name ${exp_name} --hyp ${hyp_cfg} --test_freq 2 --save_freq 20 --rad_id ${rad_id} --fold ${FOLD}
done

rad_id=all
for FOLD in 3;do
  exp_name=yolov4_p6_${rad_id}F${FOLD}

  echo training with ${rad_id} fold ${FOLD}
  python -m torch.distributed.launch --nproc_per_node ${GPUS} train.py --batch-size ${BATCH_SIZE} --epochs 100 \
  --img-size ${img_size} ${img_size} --data ${data_cfg} --cfg ${model_cfg} --weights ${pretrained} --sync-bn \
  --device 0,1,2,3,4,5,6,7 --name ${exp_name} --hyp ${hyp_cfg} --test_freq 2 --save_freq 30 --rad_id ${rad_id} --fold ${FOLD}
done


data_cfg=vinbigdata_configs/data/vinbigdata.yaml
rad_id=all
for FOLD in 1 3;do
  exp_name=yolov4_p6_MyOld_${rad_id}F${FOLD}

  echo training with ${rad_id} fold ${FOLD}
  python -m torch.distributed.launch --nproc_per_node ${GPUS} train.py --batch-size ${BATCH_SIZE} --epochs 100 \
  --img-size ${img_size} ${img_size} --data ${data_cfg} --cfg ${model_cfg} --weights ${pretrained} --sync-bn \
  --device 0,1,2,3,4,5,6,7 --name ${exp_name} --hyp ${hyp_cfg} --test_freq 2 --save_freq 20 --rad_id ${rad_id} --fold ${FOLD}
done


img_size=1536
data_cfg=vinbigdata_configs/data/vinbigdata.yaml
model_cfg=vinbigdata_configs/model/yolov4-p7_vinbig.yaml
pretrained='data/yolov4-p7.pt'

rad_id=all
for FOLD in 1;do
  exp_name=yolov4_p7_MyOld_${rad_id}F${FOLD}

  echo training with ${rad_id} fold ${FOLD}
  python -m torch.distributed.launch --nproc_per_node ${GPUS} train.py --batch-size ${BATCH_SIZE} --epochs 100 \
  --img-size ${img_size} ${img_size} --data ${data_cfg} --cfg ${model_cfg} --weights ${pretrained} --sync-bn \
  --device 0,1,2,3,4,5,6,7 --name ${exp_name} --hyp ${hyp_cfg} --test_freq 2 --save_freq 20 --rad_id ${rad_id} --fold ${FOLD}
done