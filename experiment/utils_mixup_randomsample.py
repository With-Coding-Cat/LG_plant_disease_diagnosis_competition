from torch.utils.data import Dataset
import torch
import cv2
import pandas as pd
import numpy as np
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from meta_data import disease_mask, risk_with_crop_mask, disease_encoding
import json

class TrainValDataset(Dataset):
    def __init__(self, data_path, train=True, target_label=None, crop_list=None, disease_list=None, mixup=False, mixup_alpha=None, mixup_beta=None):
        super().__init__()
        self.df1 = pd.read_csv(data_path)
        self.mixup = mixup
        if mixup:
            #self.df2 = pd.read_csv(data_path)
            #self.df2 = self.df1.sample(frac=1).reset_index(drop=True)
            #self._shuffle_dataframe()
            # self.mixup_alpha = mixup_alpha
            # self.mixup_beta = mixup_beta
            self.beta_distribution = torch.distributions.Beta(mixup_alpha, mixup_beta)
        
        if target_label is None:
            print('Check label.')
            exit()
        self.label = target_label
        self.crop_list = crop_list
        self.disease_list = disease_list
        self.train = train
        if train and mixup:
            print("Use MixIn Augmentation")
            self.augmentation = train_aug_without_normalization()
            self.normalization = normalization_aug()
        elif train:
            print("Use Train Augmentation")
            self.augmentation = train_aug()
        else:
            print("Use Test Augmentation")
            self.augmentation = test_aug()
            
    def __len__(self):
        return len(self.df1)
    
    def __getitem__(self, index):
        target_line1 = self.df1.iloc[index, :]
        if self.mixup:
            target_line2 = self.df2.iloc[index, :]
            img0 = image_load_and_agmentation(target_line1["img_path"], self.augmentation, ordinate=target_line1['ordinate'])
            img1 = image_load_and_agmentation(target_line2["img_path"], self.augmentation, ordinate=target_line2['ordinate'])
            beta_value = self.beta_distribution.sample()
            # beta_value = np.random.beta(self.mixup_alpha, self.mixup_beta)
            img = img0 * beta_value.numpy() + img1 * (1-beta_value).numpy()
            img = self.normalization(image=img)['image']
            return img, torch.tensor(target_line1['crop']),  torch.tensor(target_line1['disease']), torch.tensor(target_line1['risk']), torch.tensor(disease_mask[target_line1['crop']]), torch.tensor(risk_with_crop_mask[10*target_line1['disease']+target_line1['crop']]),\
                torch.tensor(target_line2['crop']),  torch.tensor(target_line2['disease']), torch.tensor(target_line2['risk']), torch.tensor(disease_mask[target_line2['crop']]), torch.tensor(risk_with_crop_mask[10*target_line2['disease']+target_line2['crop']]), beta_value
            
        elif self.train:
            img = image_load_and_agmentation(target_line1["img_path"], self.augmentation, ordinate=target_line1['ordinate'])
        else:
            if self.label in ('crop', 'total'):
                img = image_load_and_agmentation(target_line1["img_path"], self.augmentation)
            else:
                img = torch.tensor(-1)
        
        if self.label == 'crop':
            return img#, torch.tensor(target_line['crop'])
        elif self.label == 'disease':
            #return img, torch.tensor(target_line['disease']), torch.tensor(disease_mask[target_line['crop']])
            return img, torch.tensor(disease_mask[target_line1['crop']]) #torch.tensor(target_line['disease']), torch.tensor(disease_mask[target_line['crop']])
        elif self.label == 'risk':
            return img, torch.tensor(target_line1['disease']), torch.tensor(risk_with_crop_mask[10*target_line1['disease']+target_line1['crop']]) #torch.tensor(target_line['risk']), 
        elif self.label == 'total':
            # img, crop, disease, risk, disease_mask, risk_mask
            return img, torch.tensor(target_line1['crop']),  torch.tensor(target_line1['disease']), torch.tensor(target_line1['risk']), torch.tensor(disease_mask[target_line1['crop']]), torch.tensor(risk_with_crop_mask[10*target_line1['disease']+target_line1['crop']])

    def _change_label_and_update_crop(self, crop_list):
        self.df['crop'] = crop_list
        self.label = 'disease'
        
    def _change_label_and_update_disease(self, disease_list):
        self.df['disease'] = disease_list
        self.label = 'risk'
        
    def _shuffle_dataframe(self):
        self.df2 = self.df1.sample(frac=1).reset_index(drop=True) 


def image_load_and_agmentation(img_path, augmantation, ordinate=None):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if ordinate:
        ordinate = json.loads(ordinate)
        aug_dict = augmantation(image=img, bboxes=[ordinate+['test']])
    else:
        aug_dict = augmantation(image=img)
    return aug_dict['image']
    
# def train_aug():
#     return albu.Compose([
#         albu.Flip(),
#         albu.Resize(448, 448, always_apply=True),
#         albu.Rotate(limit=90, always_apply=True),
#         albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
#         ToTensorV2()
#     ])
    
def train_aug():
    return albu.Compose([
        albu.Flip(),
        albu.Rotate(limit=90, always_apply=True),
        albu.RandomSizedBBoxSafeCrop(384, 384, always_apply=True), #384, 384, #448, 384
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0., always_apply=True),
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
        ToTensorV2(),
    ], bbox_params=albu.BboxParams(format='coco'))
    
def train_aug_without_normalization():
    return albu.Compose([
        albu.Flip(),
        albu.Rotate(limit=90, always_apply=True),
        albu.RandomSizedBBoxSafeCrop(384, 384, always_apply=True), #384, 384, #448, 384
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0., always_apply=True),
        #albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
        #ToTensorV2(),
    ], bbox_params=albu.BboxParams(format='coco'))
    
def normalization_aug():
    return albu.Compose([
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True),
        ToTensorV2(),
    ])

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
