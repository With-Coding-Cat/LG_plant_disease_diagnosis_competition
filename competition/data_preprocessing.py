import json
import os
import time
import argparse
from multiprocessing import Pool
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split, StratifiedKFold

from meta_data import use_columns, disease_encoding

def prepare_dataframe(csv_files, json_files, img_files, save_path, percentile_num=50):
    csv_files.sort()
    if json_files[0] is not None:
        json_files.sort()
    img_files.sort()
    files_list_and_percentile_num = [(csv_file, json_file, img_file, percentile_num) for csv_file, json_file, img_file in zip(csv_files, json_files, img_files)]
    
    start_time = time.time()
    print('Data Loading.')
    with Pool() as pool:
        dict_list = pool.map(return_dict_from_files, files_list_and_percentile_num)
    
    df = pd.DataFrame(dict_list)
    df.to_csv(save_path, index=False)
    print(f"Data Loading Complete. Time Spend: {time.time() - start_time:.4} seconds")
    
            
def make_nan(x):
    if str(x) == '-':
        return np.nan
    return x

def return_dict_from_files(files_list_and_percentile_num: tuple):
    csv_file, json_file, img_file, percentile_num = files_list_and_percentile_num
    df_dict = {}
    
    temp = pd.read_csv(csv_file)
    temp = temp[use_columns]
    temp = temp.applymap(make_nan).astype(np.float64).describe(percentiles=[i/percentile_num for i in range(percentile_num+1)])
    temp_percentiles = temp.iloc[4:-1]
    
    for column in use_columns:
        if "누적일사" in column:
            if temp.loc['count', column] == 0:
                df_dict[column+'_is_exist'] = 0
            else:
                df_dict[column+'_is_exist'] = 1
            if np.isnan(temp_percentiles.loc['0%', column]):
                for i in range(len(temp_percentiles)):
                    df_dict[column+f'_{i}'] = -1
            else:
                for i in range(len(temp_percentiles)):
                    df_dict[column+f'_{i}'] = temp_percentiles.loc[:, column][i]
                    
        elif "CO2" in column:
            if temp.loc['count', column] == 0:
                df_dict[column+'_is_exist'] = 0
            else:
                df_dict[column+'_is_exist'] = 1
            if "최고" in column:
                if np.isnan(temp.loc['max', column]):
                    df_dict[column+'_max'] = -1
                else:
                    df_dict[column+'_max'] = temp.loc['max', column]
            elif "최저" in column:
                if np.isnan(temp.loc['min', column]):
                    df_dict[column+'_min'] = -1
                else:
                    df_dict[column+'_min'] = temp.loc['min', column]
            else:
                if np.isnan(temp_percentiles.loc['0%', column]):
                    for i in range(len(temp_percentiles)):
                        df_dict[column+f'_{i}'] = -1
                else:
                    for i in range(len(temp_percentiles)):
                        df_dict[column+f'_{i}'] = temp_percentiles.loc[:, column][i]
        
        else:
            if "최고" in column:
                df_dict[column+'_max'] = temp.loc['max', column]
            elif "최저" in column:
                df_dict[column+'_min'] = temp.loc['min', column]
            else:
                for i in range(len(temp_percentiles)):
                    df_dict[column+f'_{i}'] = temp_percentiles.loc[:, column][i]
    
    
    df_dict['img_path'] = img_file
    if json_file:
        with open(json_file, 'r', encoding='utf-8') as json_f:
            temp_dict = json.load(json_f)
        
            df_dict['crop'] = temp_dict['annotations']['crop'] - 1
            
            if df_dict['crop'] == 0 or df_dict['crop'] == 3:
                df_dict['disease'] = disease_encoding.index('0')
            else:
                df_dict['disease'] = disease_encoding.index(temp_dict['annotations']['disease'])
            df_dict['risk'] = temp_dict['annotations']['risk']
            xyhw = temp_dict['annotations']['bbox'][0]
            coordinate = [int(xyhw['x']), int(xyhw['y']), int(xyhw['w']), int(xyhw['h'])]
            df_dict['coordinate'] = coordinate
            df_dict['strat'] = str(temp_dict['annotations']['crop']) + str(temp_dict['annotations']['disease']) + str(temp_dict['annotations']['risk']) + str(temp_dict['annotations']['area']) + str(temp_dict['annotations']['grow'])
            
    return df_dict

def kfold_split_save(csv_file, save_folder, n_splits=5, startified='strat'):
    raw_data = pd.read_csv(csv_file)
    target_data = raw_data.drop(columns=['strat'])
    strat = raw_data[[startified]]
    kfolds = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for i, (train_index, test_index) in enumerate(kfolds.split(target_data, strat)):
        X_train, X_test = target_data.iloc[train_index, :], target_data.iloc[test_index, :]
        X_train.to_csv(os.path.join(save_folder, f'{i}_train.csv'), index=False)
        X_test.to_csv(os.path.join(save_folder, f'{i}_test.csv'), index=False)
        
def strat_split_save(csv_file, save_folder, test_size=0.1, stratified='strat'):
    raw_data = pd.read_csv(csv_file)
    strat = raw_data[[stratified]] #strat
    target_data = raw_data.drop(columns=stratified)
    train_set, test_set, _, _ = train_test_split(target_data, strat, test_size=test_size, stratify=strat)
    
    train_set.to_csv(os.path.join(save_folder, 'train.csv'), index=False)
    test_set.to_csv(os.path.join(save_folder, 'test.csv'), index=False)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select Preprocess Mode')
    parser.add_argument('--task', type=str, required=True, choices=['train', 'test'], help='Select the task')
    parser.add_argument('--data-folder', type=str, default='/data', help='Data folder path')
    parser.add_argument('--processed-csv', type=str, default='processed_train.csv', help='CSV file name to be saved')
    # parser.add_argument('--kfold-save-folder', type=str, default='kfold_dataset', help='kfold dataset folder in csv form')
    parser.add_argument('--stratified-save-folder', type=str, default='stratified_dataset', help='stratified dataset folrder in csv form')
    parser.add_argument('--test-size', type=float, default=0.1, help='Test size to be splitted')
    parser.add_argument('--persentile-num', type=int, default=50, help='How many percentile score to use in preprocessing')
    args = parser.parse_args()
    
    if args.task == 'train':
        print('Preprocessing for train.')
        csv_files = glob(os.path.join(args.data_folder, 'train', '*', '*.csv'))
        json_files = glob(os.path.join(args.data_folder, 'train', '*', '*.json'))
        img_files = glob(os.path.join(args.data_folder, 'train', '*', '*.jpg'))
        # os.makedirs(args.kfold_save_folder, exist_ok=True)
        os.makedirs(args.stratified_save_folder, exist_ok=True)
        
        prepare_dataframe(csv_files, json_files, img_files, args.processed_csv, percentile_num=args.persentile_num)
        # kfold_split_save(args.processed_csv, args.kfold_save_folder, n_splits=10, startified='strat')
        strat_split_save(args.processed_csv, args.stratified_save_folder, test_size=args.test_size, stratified='strat')
        
    elif args.task == 'test':
        print('Preprocessing for test.')
        csv_files = glob(os.path.join(args.data_folder, 'test', '*', '*.csv'))
        json_files = [None] * len(csv_files)
        img_files = glob(os.path.join(args.data_folder, 'test', '*', '*.jpg'))
        
        prepare_dataframe(csv_files, json_files, img_files, args.processed_csv, percentile_num=args.persentile_num)