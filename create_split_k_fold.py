import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import argparse


def main(config):

    np.random.seed(config.seed)

    file_meta = "EMVD/metadata_files.csv"

    #read df and filter
    df_meta = pd.read_csv(file_meta)
    df_meta = df_meta[df_meta['type']=='Technique']
    df_meta = df_meta[df_meta['authors_rank']!='0']
    df_meta = df_meta[df_meta['name']!='GrindInhale']
    df_meta = df_meta.sort_values(by=['singer_id'])
    n_splits = config.n_splits

    ######################
    ########## K FOLDS
    ######################

    df_meta = df_meta.reset_index(drop=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    split_gen = kf.split(df_meta.index.values.tolist())
    train_splits = []
    valid_splits = []
    eval_splits = []

    for idx, (fulltrain_index, eval_index) in enumerate(split_gen):
        train_samples = int(0.7 * len(fulltrain_index))
        train_index, valid_index = train_test_split(fulltrain_index, train_size=train_samples, random_state=0)

        train_splits.append(train_index)
        valid_splits.append(valid_index)
        eval_splits.append(eval_index)

    for split_idx in range(n_splits):
        split_name = f'split{split_idx}'
        
        df_meta[split_name] = 'train'
        df_meta.loc[valid_splits[split_idx], split_name] = 'valid'
        df_meta.loc[eval_splits[split_idx], split_name] = 'eval'


    #######################
    ###################
    # SAVE SPLIT
    #################
    #####################

    df = pd.read_csv(file_meta)

    # Merge the DataFrames on the 'file_name' column using a left join
    merged_df = df.merge(df_meta.drop(columns=['singer_id', 'type', 'name', 'range', 'vowel', 'authors_rank', 'duration(s)']), on='file_name', how='left')
    merged_df = merged_df.drop(columns=['singer_id', 'type', 'name', 'range', 'vowel', 'authors_rank', 'duration(s)'])
    # Replace NaN values with 'None'
    merged_df = merged_df.fillna('-')

    # Save the merged DataFrame as an Excel file
    merged_df.to_csv('new_split_kfolds.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', '--seed', help='chosen seed for numpy', default=0)
    parser.add_argument('-n_splits', '--n_splits', help='number of splits for the cross-validation', default=4)
    config = parser.parse_args()
    main(config)
