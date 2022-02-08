import os
import argparse
import glob
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from meta_data import disease_encoding

parser = argparse.ArgumentParser(description='Select Train Mode')
parser.add_argument('--target', type=str, required=True, choices=['crop', 'disease', 'risk'], help='Select target label.')
parser.add_argument('--save-folder', type=str, default=None, help='Save folder for Model. Recommand Not To Use.')
parser.add_argument('--dataset-folder', type=str, default='stratified_dataset', help='Dataset folder made by data preprocessing')
parser.add_argument('--early-stop', type=int, default=1000)
args = parser.parse_args()

BASE_SAVE_FOLDER = 'model_checkpoint/boosting'
EARLY_STOP_COUNT = args.early_stop

target_label = args.target
save_folder = args.save_folder
dataset_folder = args.dataset_folder
if save_folder:
    os.makedirs(save_folder, exist_ok=True)
else:
    if target_label in ('crop', 'disease', 'risk'):
        save_folder = os.path.join(BASE_SAVE_FOLDER, target_label)
        os.makedirs(save_folder, exist_ok=True)
    else:
        print("Check the label name")
        exit()

if target_label == 'crop':
    total_labels = [i for i in range(6)]
    ignored_features = ['img_path', 'coordinate']
    cat_features = ['외부 누적일사 평균_is_exist', '내부 CO2 평균_is_exist','내부 CO2 최고_is_exist','내부 CO2 최저_is_exist']
    drop_labels = ['crop', 'disease', 'risk']
elif target_label == 'disease':
    total_labels = [i for i in range(len(disease_encoding))]
    ignored_features = ['img_path', 'coordinate']
    cat_features = ['외부 누적일사 평균_is_exist', '내부 CO2 평균_is_exist','내부 CO2 최고_is_exist','내부 CO2 최저_is_exist', 'crop']
    drop_labels = ['disease', 'risk']
elif target_label == 'risk':
    total_labels = [i for i in range(4)]
    ignored_features = ['img_path', 'coordinate']
    cat_features = ['외부 누적일사 평균_is_exist', '내부 CO2 평균_is_exist','내부 CO2 최고_is_exist','내부 CO2 최저_is_exist', 'crop', 'disease']
    drop_labels = ['risk']
    
train_csv_files = sorted(glob.glob(dataset_folder + '/*train.csv'))
test_csv_files = sorted(glob.glob(dataset_folder + '/*test.csv'))

for k_fold, (train_csv, test_csv) in enumerate(zip(train_csv_files, test_csv_files)):
    train_df = pd.read_csv(train_csv)
    train_df = train_df.drop(columns=ignored_features)
    train_x, train_y = train_df.drop(columns=drop_labels), train_df[target_label]
    
    test_df = pd.read_csv(test_csv)
    test_df = test_df.drop(columns=ignored_features)
    test_x, test_y = test_df.drop(columns=drop_labels), test_df[target_label]
    
    if target_label == 'crop':
        border_count = [7]
        random_strength = [1]
        leaf_estimation_iterations = [None]
        learning_rate = [0.1]

    elif target_label == 'disease':
        border_count = [64]
        random_strength = [2]
        leaf_estimation_iterations = [None]
        learning_rate = [0.1]

    elif target_label == 'risk':
        border_count = [64]
        random_strength = [2]
        leaf_estimation_iterations = [None]
        learning_rate = [0.1]
        
    grids = [(leaf, learn, streng, border) for leaf in leaf_estimation_iterations for learn in learning_rate for streng in random_strength for border in border_count]
    
    max_acc = -float('inf')
    max_f1 = -float('inf')
    
    for leaf, learn, streng, border in tqdm(grids):
        model = CatBoostClassifier(
            iterations = 1000000,
            task_type = 'GPU',
            devices='0',
            thread_count = 16,
            learning_rate = learn,
            border_count=border,
            random_strength=streng,
            leaf_estimation_iterations=leaf,
            auto_class_weights='Balanced',
            max_depth=8,
            eval_metric='TotalF1',
            use_best_model=True,
            cat_features=cat_features,
            verbose=True,
        )
        model.fit(
            train_x, train_y,
            eval_set = [(test_x, test_y)],
            early_stopping_rounds=EARLY_STOP_COUNT
        )
        
        preds = model.predict(test_x)
        res = model.score(test_x, test_y)
        report = classification_report(test_y, preds)
        f1_score_result = f1_score(test_y, preds, average='macro', labels=total_labels)
        print(report)
        preds_train = model.predict(train_x)
        report = classification_report(train_y, preds_train)
        print( '-' *100)
        print(report)
        print(f'{target_label}--F1 Score:', f1_score_result, '\tAcc:', res, '\tBest F1 Score:', max(max_f1, f1_score_result), '\tBest Acc:', max(max_acc, res))
        
    ### from here, it was for grid search.
        # if max_acc < res:
        #     max_acc = res
        #     # max_acc_leaf = leaf
        #     # max_acc_learn = learn
        #     # max_acc_streng = streng
        #     # max_acc_border = border
        #     model.save_model(os.path.join(save_folder, f'max_acc_{k_fold}.pkl'))
        # if max_f1 < f1_score_result:
        #     max_f1 = f1_score_result
        #     # max_f1_leaf = leaf
        #     # max_f1_learn = learn
        #     # max_f1_streng = streng
        #     # max_f1_border = border
        #     model.save_model(os.path.join(save_folder, f'max_f1_{k_fold}.pkl'))
        
    #     with open(f'catboost_{target_label}_log.txt', 'a', encoding='utf-8') as f:
    #         f.write('-'*60)
    #         f.write('\n')
    #         f.write(f'leaf_estimation_iterations: {leaf}\t learn_rate: {learn}\t random_strength: {streng}\t border_count: {border}\n')
    #         f.write(report)
    #         f.write('\n')
    #         f.write(f'F1_score: {f1_score_result}\tAcc: {res}\n')
    #         f.write('-'*60)
            
    # with open(f'catboost_{target_label}_log.txt', 'a', encoding='utf-8') as f:
    #         f.write('-'*60)
    #         f.write('Result\n')
    #         f.write(f'Max Acc: {max_acc}\n')
    #         f.write(f'leaf_estimation_iterations: {max_acc_leaf}\t learn_rate: {max_acc_learn}\t random_strength: {max_acc_streng}\t border_count: {max_acc_border}\n')
    #         f.write('-'*60)
    #         f.write('Result\n')
    #         f.write(f'Max F1: {max_f1}\n')
    #         f.write(f'leaf_estimation_iterations: {max_f1_leaf}\t learn_rate: {max_f1_learn}\t random_strength: {max_f1_streng}\t border_count: {max_f1_border}\n')
            
        
    model.save_model(os.path.join(save_folder, f'{target_label}_{k_fold}.pkl'))
    # Here is to check feature importance
    # importance = model.feature_importances_
    # col_names = train_x.columns
    # importance_with_name = [(impor, name) for name, impor in zip(col_names, importance)]
    # for i in sorted(importance_with_name):
    #    print(i)
    #exit()