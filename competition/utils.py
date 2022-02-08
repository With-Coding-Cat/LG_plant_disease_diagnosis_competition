import json
import cv2
import pandas as pd
import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset
from meta_data import disease_mask, risk_with_crop_mask

class TrainValDataset(Dataset):
    def __init__(self, data_path, train=True, target_label=None, crop_list=None, disease_list=None):
        super().__init__()
        self.df = pd.read_csv(data_path)
        if target_label is None:
            print('Check label.')
            exit()
        self.label = target_label
        self.crop_list = crop_list
        self.disease_list = disease_list
        self.train = train
        if train:
            print("Use Train Augmentation")
            self.augmentation = train_aug()
        else:
            print("Use Test Augmentation")
            self.augmentation = test_aug()
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        target_line = self.df.iloc[index, :]
        if self.train:
            img = image_load_and_agmentation(target_line["img_path"], self.augmentation, ordinate=target_line['coordinate'])
        else:
            if self.label in ('crop', 'total'):
                img = image_load_and_agmentation(target_line["img_path"], self.augmentation)
        
        if self.label == 'crop':
            return img
        
        elif self.label == 'disease':
            return torch.tensor(disease_mask[target_line['crop']])
        
        elif self.label == 'risk':
            return torch.tensor(target_line['disease']), torch.tensor(risk_with_crop_mask[10*target_line['disease']+target_line['crop']]) 
        
        elif self.label == 'total':
            return img, torch.tensor(target_line['crop']),  torch.tensor(target_line['disease']), torch.tensor(target_line['risk']), torch.tensor(disease_mask[target_line['crop']]), torch.tensor(risk_with_crop_mask[10*target_line['disease']+target_line['crop']])

    def _change_label_and_update_crop(self, crop_list):
        self.df['crop'] = crop_list
        self.label = 'disease'
        
    def _change_label_and_update_disease(self, disease_list):
        self.df['disease'] = disease_list
        self.label = 'risk'

def image_load_and_agmentation(img_path, augmantation, ordinate=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if ordinate:
        ordinate = json.loads(ordinate)
        aug_dict = augmantation(image=img, bboxes=[ordinate+['test']])
    else:
        aug_dict = augmantation(image=img)
    return aug_dict['image']
    
    
def train_aug():
    return albu.Compose([
        albu.Flip(),
        albu.Rotate(limit=90, always_apply=True),
        albu.RandomSizedBBoxSafeCrop(384, 384, always_apply=True),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0., always_apply=True),
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
        ToTensorV2(),
    ], bbox_params=albu.BboxParams(format='coco'))

def test_aug():
    return albu.Compose([
        albu.Resize(384, 384, always_apply=True),
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
        ToTensorV2()
    ])
    
    
def train_aug_deit():
    return albu.Compose([
        albu.Flip(),
        albu.Rotate(limit=90, always_apply=True),
        albu.RandomSizedBBoxSafeCrop(384, 384, always_apply=True), #384, 384, #448, 384
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0., always_apply=True),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
        ToTensorV2(),
    ], bbox_params=albu.BboxParams(format='coco'))

def test_aug_deit():
    return albu.Compose([
        albu.Resize(384, 384, always_apply=True),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True),
        ToTensorV2()
    ])
    
def total_acc_cal(preds_crop, answers_crop, preds_disease, answers_disease, preds_risk, answers_risk):
    preds_crop = np.array(preds_crop)
    answers_crop = np.array(answers_crop)
    preds_disease = np.array(preds_disease)
    answers_disease = np.array(answers_disease)
    preds_risk = np.array(preds_risk)
    answers_risk = np.array(answers_risk)
    
    crop_array = preds_crop == answers_crop
    disease_array = preds_disease == answers_disease
    risk_array = preds_risk == answers_risk
    
    acc_count = crop_array.astype(np.int8) + disease_array.astype(np.int8) + risk_array.astype(np.int8)
    return np.sum(acc_count == 3) / len(acc_count)
    
    
def get_sampler_weight(df: pd.DataFrame or pd.Series) -> list:
    count_dict = dict(df.value_counts())
    df_list = df.to_list()
    weight = [1./count_dict[target] for target in df_list]
    return weight

def return_disease_mask(crop_list, DEVICE):
    disease_mask_tensor = []
    for crop in crop_list:
        disease_mask_tensor.append(disease_mask[crop])
    disease_mask_tensor = torch.tensor(disease_mask_tensor, device=DEVICE)
    

def return_risk_mask(crop_list, disease_list, DEVICE):
    risk_mask_tensor = []
    for crop, disease in zip(crop_list, disease_list):
        risk_mask_tensor.append(risk_with_crop_mask(crop + disease*10))
    risk_mask_tensor = torch.tensor(risk_mask_tensor, device=DEVICE)
    
    
def ensemble_cal(DL_pro_target_df_catboost):
    DL_pro, target_df, boosting = DL_pro_target_df_catboost
    boosting_pro = boosting.predict_proba(target_df)
    return np.argmax(DL_pro + boosting_pro, axis=-1).tolist()
    
def ensemble_cal_with_mask(DL_pro_target_df_catboost):
    DL_pro, target_df, boosting, mask = DL_pro_target_df_catboost
    boosting_pro = boosting.predict_proba(target_df)
    boosting_pro = np.ma.MaskedArray(data=boosting_pro, mask=mask.numpy())
    return np.argmax(DL_pro + boosting_pro.data, axis=-1).tolist()