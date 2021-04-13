import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from itertools import zip_longest
import os
import torch


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return np.array([list(i) for i in zip_longest(*args, fillvalue=fillvalue)])


df_cls = pd.read_csv(
    f'../results/multilabel_cls/v1/multilabel_efnb4_v1_fold0_preds.csv'
).set_index('image_id')

for fold in range(1, 5):
    df_tmp = pd.read_csv(
        f'../results/multilabel_cls/v1/multilabel_efnb4_v1_fold{fold}_preds.csv'
    ).set_index('image_id')
    assert (df_cls.index == df_tmp.index).all()
    df_cls += df_tmp

df_cls = (df_cls / 5).reset_index()
df_cls = df_cls.drop('cls11_ext', 1)


df_cls2 = pd.read_csv(
    f'../results/multilabel_cls/v3/multilabel_efnb4_fold0_preds.csv'
).set_index('image_id')

for fold in range(1, 5):
    df_tmp = pd.read_csv(
        f'../results/multilabel_cls/v3/multilabel_efnb4_fold{fold}_preds.csv'
    ).set_index('image_id')
    assert (df_cls2.index == df_tmp.index).all()
    df_cls2 += df_tmp

df_cls2 = (df_cls2 / 5).reset_index()


df_cls = (
        0.5 * df_cls.set_index('image_id').sort_index() + 0.5 * df_cls2.set_index('image_id').sort_index()
).reset_index()


# b4 1024 AUCs
aucs = np.array([
    0.97000972, 0.97524663, 0.95992002, 0.97731252, 0.98552718,
    0.98639051, 0.98022454, 0.95938934, 0.96390168, 0.90608034,
    0.9893505, 0.94720878, 0.99756838, 0.97154021, 0.99229729
])
aucs_13 = aucs[:-1]


# b4 1280 AUCs
aucs = np.array([
    0.98198037, 0.97867319, 0.97409582, 0.98339185, 0.98689603,
    0.98672216, 0.98150991, 0.96382269, 0.9698253, 0.92592461,
    0.99223501, 0.95780923, 0.9977424, 0.97797093, 0.99369823,
])
aucs_13 = 0.5 * aucs_13 + 0.5 * aucs[:-1]
quotas = np.round(aucs_13 / aucs_13.sum() * 1912 * 14)


for i, quota in zip(np.arange(0, 14).astype(str), quotas):
    #     if i in ['3','5']: # for cls3 and cls5 use 1-cls14 classifier score as multiplier (better CV + LB)
    #         df_cls[i] = 1-df_cls['14']
    df_cls.loc[df_cls[i].rank() > quota, i] = 1

df_cls = pd.melt(df_cls, id_vars='image_id').rename(columns={'variable': 'yolo_class'})

df_cls['yolo_class'] = df_cls['yolo_class'].astype(int)


# df_sub_b4_cls = pd.read_csv('../results/results_test_cls_0.278_withoutClassifier.csv')
df_sub_b4_cls = pd.read_csv('../results/merge_results_thr0.001.csv')
d = df_sub_b4_cls.set_index('image_id')['PredictionString'].str.split().apply(grouper, n=6).to_dict()
df_main = pd.concat(
    [pd.DataFrame(
        d[key], columns=['yolo_class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max']
    ) for key in d.keys()]
)
img_ids = [[key] * len(item) for key, item in d.items()]
df_main['image_id'] = [item for sublist in img_ids for item in sublist]

for i in ['confidence', 'x_min', 'y_min', 'x_max', 'y_max']:
    df_main[i] = df_main[i].astype(float)
df_main['yolo_class'] = df_main['yolo_class'].astype(int)

classifier_scores = df_cls.loc[
    df_cls['yolo_class'] == 14, ['image_id', 'value']
].reset_index(drop=True).rename(
    columns={'value': 'cls_score_mean'}
)
yolo_dat = df_main.merge(
    classifier_scores,
    on='image_id',
    how='outer'
).merge(
    df_cls,
    on=['image_id', 'yolo_class'],
    how='left'
)

yolo_dat['confidence'] = yolo_dat['confidence'] * yolo_dat['value']

yolo_dat['PredictionString'] = \
    yolo_dat['yolo_class'].astype(str) + " " + yolo_dat['confidence'].astype(str) + " " + \
    yolo_dat['x_min'].astype(str) + " " + yolo_dat['y_min'].astype(str) + " " + \
    yolo_dat['x_max'].astype(str) + " " + yolo_dat['y_max'].astype(str)

yolo_predstr = yolo_dat.groupby('image_id', as_index=False)['PredictionString'].apply(lambda x: " ".join(x))

final_output = yolo_predstr.merge(
    classifier_scores,
    on='image_id',
    how='outer'
)

final_output['PredictionString'] = final_output['PredictionString'] + ' 14 ' + final_output['cls_score_mean'].astype(
    str) + ' 0 0 1 1'
final_output = final_output[['image_id', 'PredictionString']]
final_output.to_csv(f'../results/0.280_aortic_cls10_multimul_multithres13.csv', index=False)





