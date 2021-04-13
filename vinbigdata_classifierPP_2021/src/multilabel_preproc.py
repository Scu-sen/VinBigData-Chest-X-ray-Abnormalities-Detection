import random
import numpy as np
import pandas as pd
from collections import Counter, defaultdict


def main():
    input_path = '../input/vinbigdata-chest-xray-abnormalities-detection/'
    
    # read train.csv
    df = pd.read_csv(input_path+"train.csv")
    
    # Divide into 5 folds using stratified group-k-fold
    for fold_ind, (train_ind, val_ind) in enumerate(stratified_group_k_fold(
        _, np.array(df['class_id']), np.array(df['image_id']), 5, seed=42
    )):
        df.loc[val_ind,'fold'] = fold_ind

    # save the folds
    df.to_csv(input_path+"train_5folds.csv",index=False)

    df = pd.read_csv(input_path+'train_5folds.csv')

    # Create df for multilabel training
    df = df.groupby(['fold','image_id','class_id'], as_index=False)['rad_id'].nunique().pivot_table(
        columns='class_id', values='rad_id', index=['fold','image_id']
    ).fillna(0)/3

    # Save the df for training
    df.reset_index().to_csv(input_path+'multilabel_cls_train.csv',index=False)

    # Calculate and save the pos_weight
    np.save(input_path+'multilabel_pos_weight.npy',((df.shape[0]-df.sum()) / df.sum()).values)
    
    
def stratified_group_k_fold(X, y, groups, k, seed=None):
    """
    stratified version of Group-k-fold splitting
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

    
if __name__=='__main__':
    main()   