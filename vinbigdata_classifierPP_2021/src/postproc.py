import pandas as pd
import numpy as np
from itertools import zip_longest
import os

cls_dfs_path = '../classifier_preds/'
detector_preds_path = '../detector_preds/'
variants = ['1024', '1024_fixAR', '1280_fixAR']
weights = [0.5, 0.25, 0.25]
center_images = pd.read_csv('../test_center/test_0.139000.csv')['file'].unique() 
save_path = '../submissions/'
    
    
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return np.array([list(i) for i in zip_longest(*args, fillvalue=fillvalue)])


def main():
    ############################## Read Classifier Predictions ##############################
    cls_dfs = []
    for variant in variants:
        df_cls = pd.read_csv(
            cls_dfs_path+f'{variant}/multilabel_efnb4_fold0_preds.csv'
        ).set_index('image_id')

        for fold in range(1,5):
            df_tmp = pd.read_csv(
                cls_dfs_path+f'{variant}/multilabel_efnb4_fold{fold}_preds.csv'
            ).set_index('image_id')
            assert (df_cls.index == df_tmp.index).all()
            df_cls += df_tmp

        df_cls = (df_cls/5)
        cls_dfs.append(df_cls.sort_index())

    df_cls = sum([i*j for i,j in zip(cls_dfs,weights)]).reset_index()

    ############################## Threshold Calculation based on AUCs ##############################
    aucs1 = np.array([
        0.97000972, 0.97524663, 0.95992002, 0.97731252, 0.98552718,
        0.98639051, 0.98022454, 0.95938934, 0.96390168, 0.90608034,
        0.9893505 , 0.94720878, 0.99756838, 0.97154021
    ])

    aucs2 = np.array([
        0.98188325, 0.98017857, 0.97200959, 0.98463441, 0.9863487,
        0.9885482, 0.98159763, 0.9635907, 0.97034802, 0.92328928,
        0.9930435, 0.9592879, 0.99862277, 0.978146
    ])

    aucs3 = np.array([
        0.98155145, 0.98269187, 0.97727565, 0.98442159, 0.98678433,
        0.98895312, 0.98123528, 0.96122668, 0.96872207, 0.92257618, 
        0.9920549, 0.96089542, 0.9974569, 0.97774058
    ])

    aucs = sum([i*j for i,j in zip([aucs1,aucs2,aucs3],weights)])

    quotas = np.round(aucs/aucs.sum()*1912*14)

    ############################## Finalize Classifier Scores and Multipliers ##############################
    for i,quota in zip(np.arange(0,14).astype(str),quotas):    
        if i in ['3','5']:
            df_cls[i] = 1-df_cls['14']

        df_cls.loc[
            (df_cls[i].rank()>quota)&
            (~df_cls['image_id'].isin(center_images)),
            i
        ] = 1

    df_cls = pd.melt(df_cls, id_vars='image_id').rename(columns={'variable':'yolo_class'})

    df_cls['yolo_class'] = df_cls['yolo_class'].astype(int)

    df_cls14 = df_cls.loc[
        df_cls['yolo_class']==14,['image_id','value']
    ].reset_index(drop=True).rename(
        columns={'value':'cls14_score'}
    )

    ############################## Read Detector Predictions ##############################
    df_sub_b4_cls = pd.read_csv(detector_preds_path+'final_before_pp.csv') 
    df_aortic = pd.read_csv(detector_preds_path+'aortic_final_fixed.csv')
    df_cls10 = pd.read_csv(detector_preds_path+'final_cls10.csv')

    d = df_sub_b4_cls.set_index('image_id')['PredictionString'].str.split().apply(grouper, n=6).to_dict()
    df_main = pd.concat(
        [pd.DataFrame(
            d[key], columns=['yolo_class','confidence','x_min','y_min','x_max','y_max']
        ) for key in d.keys()]
    )
    img_ids = [[key]*len(item) for key,item in d.items()]      
    df_main['image_id'] = [item for sublist in img_ids for item in sublist]

    for i in ['confidence','x_min','y_min','x_max','y_max']:
        df_main[i] = df_main[i].astype(float)
    df_main['yolo_class'] = df_main['yolo_class'].astype(int)

    df_main = df_main.loc[
        (~df_main['yolo_class'].isin([0,10]))
    ].append(
        df_aortic.loc[df_aortic['yolo_class']==0,df_main.columns],
        ignore_index=True
    ).append(
        df_cls10.loc[df_cls10['yolo_class']==10,df_main.columns], ignore_index=True
    )

    ############################## Merge Predictions and Perform Multiplication ##############################
    df_main = df_main.merge(
        df_cls,
        on = ['image_id','yolo_class'],
        how = 'left'
    )
    df_main['confidence'] = df_main['confidence']*df_main['value']

    ############################## Format Prediction Strings ##############################
    df_main['PredictionString'] = \
    df_main['yolo_class'].astype(str) + " " + df_main['confidence'].astype(str) + " " + \
    df_main['x_min'].astype(str) + " " + df_main['y_min'].astype(str) + " " + \
    df_main['x_max'].astype(str) + " " + df_main['y_max'].astype(str)

    predstrs = df_main.groupby('image_id', as_index=False)['PredictionString'].apply(lambda x:" ".join(x))

    final_output = predstrs.merge(
        df_cls14,
        on = 'image_id',
        how = 'outer'
    )

    final_output['PredictionString'] = \
    final_output['PredictionString'] + ' 14 ' + \
    final_output['cls14_score'].astype(str) + ' 0 0 1 1'
    final_output = final_output[['image_id', 'PredictionString']]
    final_output.to_csv(save_path + f'submission.csv', index= False)
    
    
if __name__=='__main__':
    main() 