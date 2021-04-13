# directory for all the PreTrained weights
PreTrainedRoot='./PreTrainedWeights'

img_size=1280
exp_name=yolov4_p6_rad_id8F1_0129
echo "start evaluation"
export CUDA_VISIBLE_DEVICES=4,5,6,7
for SPLIT in 'test';do
  test_ims_root='data/vinbigdata/images/test'
  test_info_file='data/vinbigdata/annotations/vinbig_test.json'
  sample_sub_file=data/vinbigdata/annotations/sample_submission.csv

  out_path=inference/PreTrained_results
  weight_root=${PreTrainedRoot}/exp_${exp_name}/weights

  python inference_dbtex.py --info_file ${test_info_file} --img_root ${test_ims_root} --output ${out_path} --augment \
  --img-size ${img_size} --save_thres 1e-4 --vis_thres 0.1 --iou-thres 0.5 --num_cls 15 --sample_sub_file ${sample_sub_file} \
  --save_img_num 0 --batch_size 32 --device 4 --rect --split ${SPLIT} --post_fix best40_60_80_99Final_R8F1 \
  --weights ${weight_root}/best_${exp_name}.pt ${weight_root}/last_040.pt ${weight_root}/last_060.pt ${weight_root}/last_080.pt ${weight_root}/last_099.pt
done

exp_name=yolov4_p6_rad_id8F2_0129
echo "start evaluation"
export CUDA_VISIBLE_DEVICES=4,5,6,7
for SPLIT in 'test';do
  test_ims_root='data/vinbigdata/images/test'
  test_info_file='data/vinbigdata/annotations/vinbig_test.json'
  sample_sub_file=data/vinbigdata/annotations/sample_submission.csv

  out_path=inference/PreTrained_results
  weight_root=${PreTrainedRoot}/exp_${exp_name}/weights

  python inference_dbtex.py --info_file ${test_info_file} --img_root ${test_ims_root} --output ${out_path} --augment \
  --img-size ${img_size} --save_thres 0.0001 --vis_thres 0.1 --iou-thres 0.5 --num_cls 15 --sample_sub_file ${sample_sub_file} \
  --save_img_num 0 --batch_size 32 --device 4 --rect --split ${SPLIT} --post_fix best60_80Final_R8F2 \
  --weights ${weight_root}/best_${exp_name}.pt ${weight_root}/last_060.pt ${weight_root}/last_080.pt
done

exp_name=yolov4_p6_rad_id8F3_0130
echo "start evaluation"
export CUDA_VISIBLE_DEVICES=4,5,6,7
for SPLIT in 'test';do
  test_ims_root='data/vinbigdata/images/test'
  test_info_file='data/vinbigdata/annotations/vinbig_test.json'
  sample_sub_file=data/vinbigdata/annotations/sample_submission.csv

  out_path=inference/PreTrained_results
  weight_root=${PreTrainedRoot}/exp_${exp_name}/weights

  python inference_dbtex.py --info_file ${test_info_file} --img_root ${test_ims_root} --output ${out_path} --augment \
  --img-size ${img_size} --save_thres 0.0001 --vis_thres 0.1 --iou-thres 0.5 --num_cls 15 --sample_sub_file ${sample_sub_file} \
  --save_img_num 0 --batch_size 32 --device 7 --rect --split ${SPLIT} --post_fix best60_90Final_R8F3 \
  --weights ${weight_root}/best_${exp_name}.pt ${weight_root}/last_060.pt ${weight_root}/last_090.pt
done
exp_name=yolov4_p6_rad_id8F4_0130
echo "start evaluation"
export CUDA_VISIBLE_DEVICES=4,5,6,7
for SPLIT in 'test';do
  test_ims_root='data/vinbigdata/images/test'
  test_info_file='data/vinbigdata/annotations/vinbig_test.json'
  sample_sub_file=data/vinbigdata/annotations/sample_submission.csv

  out_path=inference/PreTrained_results
  weight_root=${PreTrainedRoot}/exp_${exp_name}/weights

  python inference_dbtex.py --info_file ${test_info_file} --img_root ${test_ims_root} --output ${out_path} --augment \
  --img-size ${img_size} --save_thres 0.0001 --vis_thres 0.1 --iou-thres 0.5 --num_cls 15 --sample_sub_file ${sample_sub_file} \
  --save_img_num 0 --batch_size 32 --device 4 --rect --split ${SPLIT} --post_fix best60_90Final_R8F4 \
  --weights ${weight_root}/best_${exp_name}.pt ${weight_root}/last_060.pt ${weight_root}/last_090.pt
done
exp_name=yolov4_p6_rad_id8F5_0130
echo "start evaluation"
export CUDA_VISIBLE_DEVICES=4,5,6,7
for SPLIT in 'test';do
  test_ims_root='data/vinbigdata/images/test'
  test_info_file='data/vinbigdata/annotations/vinbig_test.json'
  sample_sub_file=data/vinbigdata/annotations/sample_submission.csv

  out_path=inference/PreTrained_results
  weight_root=${PreTrainedRoot}/exp_${exp_name}/weights

  python inference_dbtex.py --info_file ${test_info_file} --img_root ${test_ims_root} --output ${out_path} --augment \
  --img-size ${img_size} --save_thres 0.0001 --vis_thres 0.1 --iou-thres 0.5 --num_cls 15 --sample_sub_file ${sample_sub_file} \
  --save_img_num 0 --batch_size 32 --device 4 --rect --split ${SPLIT} --post_fix best90_99Final_R8F5 \
  --weights ${weight_root}/best_${exp_name}.pt ${weight_root}/last_090.pt ${weight_root}/last_099.pt
done

exp_name=yolov4_p6_rad_id9F4_0128
echo "start evaluation"
export CUDA_VISIBLE_DEVICES=4,5,6,7
for SPLIT in 'test';do
  test_ims_root='data/vinbigdata/images/test'
  test_info_file='data/vinbigdata/annotations/vinbig_test.json'
  sample_sub_file=data/vinbigdata/annotations/sample_submission.csv

  out_path=inference/PreTrained_results
  weight_root=${PreTrainedRoot}/exp_${exp_name}/weights

  python inference_dbtex.py --info_file ${test_info_file} --img_root ${test_ims_root} --output ${out_path} --augment \
  --img-size ${img_size} --save_thres 0.0001 --vis_thres 0.1 --iou-thres 0.5 --num_cls 15 --sample_sub_file ${sample_sub_file} \
  --save_img_num 0 --batch_size 32 --device 4 --rect --split ${SPLIT} --post_fix best60_80Final_R9F4 \
  --weights ${weight_root}/best_${exp_name}.pt ${weight_root}/last_060.pt ${weight_root}/last_080.pt
done

exp_name=yolov4_p6_rad_id10F1_0129
echo "start evaluation"
export CUDA_VISIBLE_DEVICES=4,5,6,7
for SPLIT in 'test';do
  test_ims_root='data/vinbigdata/images/test'
  test_info_file='data/vinbigdata/annotations/vinbig_test.json'
  sample_sub_file=data/vinbigdata/annotations/sample_submission.csv

  out_path=inference/PreTrained_results
  weight_root=${PreTrainedRoot}/exp_${exp_name}/weights

  python inference_dbtex.py --info_file ${test_info_file} --img_root ${test_ims_root} --output ${out_path} --augment \
  --img-size ${img_size} --save_thres 0.0001 --vis_thres 0.1 --iou-thres 0.5 --num_cls 15 --sample_sub_file ${sample_sub_file} \
  --save_img_num 0 --batch_size 32 --device 4 --rect --split ${SPLIT} --post_fix best60_80Final_R10F1 \
  --weights ${weight_root}/best_${exp_name}.pt ${weight_root}/last_060.pt ${weight_root}/last_080.pt
done

img_size=1536
exp_name=yolov4_p7_0127
echo "start evaluation"
export CUDA_VISIBLE_DEVICES=4,5,6,7
for SPLIT in 'test';do
  test_ims_root='data/vinbigdata/images/test'
  test_info_file='data/vinbigdata/annotations/vinbig_test.json'
  sample_sub_file=data/vinbigdata/annotations/sample_submission.csv

  out_path=inference/PreTrained_results
  weight_root=${PreTrainedRoot}/exp_${exp_name}/weights

  python inference_dbtex.py --info_file ${test_info_file} --img_root ${test_ims_root} --output ${out_path} --augment \
  --img-size ${img_size} --save_thres 0.0001 --vis_thres 0.1 --iou-thres 0.5 --num_cls 15 --sample_sub_file ${sample_sub_file} \
  --save_img_num 0 --batch_size 32 --device 4 --rect --split ${SPLIT} --post_fix best60_80Final_p7OldRAllF1 \
  --weights ${weight_root}/best_${exp_name}.pt ${weight_root}/last_060.pt ${weight_root}/last_080.pt
done

img_size=1280
exp_name=yolov4_p6_0127
echo "start evaluation"
export CUDA_VISIBLE_DEVICES=4,5,6,7
for SPLIT in 'test';do
  test_ims_root='data/vinbigdata/images/test'
  test_info_file='data/vinbigdata/annotations/vinbig_test.json'
  sample_sub_file=data/vinbigdata/annotations/sample_submission.csv

  out_path=inference/PreTrained_results
  weight_root=${PreTrainedRoot}/exp_${exp_name}/weights

  python inference_dbtex.py --info_file ${test_info_file} --img_root ${test_ims_root} --output ${out_path} --augment \
  --img-size ${img_size} --save_thres 0.0001 --vis_thres 0.1 --iou-thres 0.5 --num_cls 15 --sample_sub_file ${sample_sub_file} \
  --save_img_num 0 --batch_size 32 --device 4 --rect --split ${SPLIT} --post_fix best60_80Final_p6OldRAllF1 \
  --weights ${weight_root}/best_${exp_name}.pt ${weight_root}/last_060.pt ${weight_root}/last_080.pt
done

img_size=1280
exp_name='yolov4_p6_0128'
for SPLIT in 'test';do
  test_ims_root='data/vinbigdata/images/test'
  test_info_file='data/vinbigdata/annotations/vinbig_test.json'
  sample_sub_file=data/vinbigdata/annotations/sample_submission.csv

  out_path=inference/PreTrained_results
  weight_root=${PreTrainedRoot}/exp_${exp_name}/weights

  python inference_dbtex.py --info_file ${test_info_file} --img_root ${test_ims_root} --output ${out_path} --augment \
  --img-size ${img_size} --save_thres 0.0001 --vis_thres 0.1 --iou-thres 0.5 --num_cls 15 --sample_sub_file ${sample_sub_file} \
  --save_img_num 0 --batch_size 32 --device 4 --rect --split ${SPLIT} --post_fix _best40_80_120Final_p6OldRAllF3 \
  --weights ${weight_root}/bestF3.pt ${weight_root}/last_040_F3.pt ${weight_root}/last_080_F3.pt ${weight_root}/last_120_F3.pt
done

img_size=1280
exp_name='yolov4_p6_allF3_0208'
for SPLIT in 'test';do
  test_ims_root='data/vinbigdata/images/test'
  test_info_file='data/vinbigdata/annotations/vinbig_test.json'
  sample_sub_file=data/vinbigdata/annotations/sample_submission.csv

  out_path=inference/PreTrained_results
  weight_root=${PreTrainedRoot}/exp_${exp_name}/weights

  python inference_dbtex.py --info_file ${test_info_file} --img_root ${test_ims_root} --output ${out_path} --augment \
  --img-size ${img_size} --save_thres 0.0001 --vis_thres 0.1 --iou-thres 0.5 --num_cls 15 --sample_sub_file ${sample_sub_file} \
  --save_img_num 0 --batch_size 32 --device 4 --rect --split ${SPLIT} --post_fix _best60_90Final_RAllF3 \
  --weights ${weight_root}/best_${exp_name}.pt ${weight_root}/last_060.pt ${weight_root}/last_090.pt
done