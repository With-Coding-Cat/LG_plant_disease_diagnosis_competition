import torch
from torch.utils.data import DataLoader
from model import BaseModel, CropHeadClassifier, DiseaseHeadClassifier, RiskHeadClassifier
from catboost import CatBoostClassifier
from utils import TrainValDataset
from meta_data import disease_encoding, disease_decoding
import argparse
import glob
import numpy as np
import re
import csv
from tqdm import tqdm

parser = argparse.ArgumentParser(description='It is for predict')
parser.add_argument('--checkpoint-torch', type=str, default='model_checkpoint/torch/multi_output/max_f1_total_0.pt', help='Checkpoint file for pytorch model', required=False)

parser.add_argument('--checkpoint-cat', type=str, default='model_checkpoint/boosting', help='Checkpoint file for pytorch model', required=False)
parser.add_argument('--csv-file', type=str, default='preprocessed_test.csv', required=False)
parser.add_argument('--csv-save', type=str, default='result.csv', required=False)

args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU_DEVICE = torch.device('cpu')
BATCH_SIZE = 120
NUM_WORKERS = 4
total_crop_labels = [i for i in range(6)]
total_disease_labels = [i for i in range(len(disease_encoding))]
total_risk_labels = [i for i in range(4)]
test_csv = args.csv_file
target_labels = ['crop', 'disease', 'risk']

print("Start Predict")
test_dataset = TrainValDataset(test_csv, train=False, target_label='crop')

base_model = BaseModel(pretrained=False).to(DEVICE)
crop_model = CropHeadClassifier().to(DEVICE)
disease_model = DiseaseHeadClassifier().to(DEVICE)
risk_model = RiskHeadClassifier().to(DEVICE)

checkpoint = torch.load(args.checkpoint_torch, map_location=DEVICE)

base_model.load_state_dict(checkpoint['base_model'])
crop_model.load_state_dict(checkpoint['crop_model'])
disease_model.load_state_dict(checkpoint['disease_model'])
risk_model.load_state_dict(checkpoint['risk_model'])
del checkpoint

crop_cat = CatBoostClassifier()
disease_cat = CatBoostClassifier()

for cat_file_path, cat_model in zip(sorted(glob.glob(args.checkpoint_cat+'/*/*')), [crop_cat, disease_cat]):
    cat_model.load_model(cat_file_path)

base_model.eval()
crop_model.eval()
disease_model.eval()
risk_model.eval()

img_tensors = []
preds_crop = []
preds_disease = []
preds_risk = []
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=BATCH_SIZE*2, drop_last=False, pin_memory=True)
for current_label in target_labels:
    if current_label != 'crop':
        if current_label == 'disease':
            test_dataset._change_label_and_update_crop(preds_crop)
        else:
            test_dataset._change_label_and_update_disease(preds_disease)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            if current_label == 'crop':
                for img_idx, data in tqdm(enumerate(test_dataloader)):
                    img = data
                    img = img.to(DEVICE)
                    img_processed = base_model(img)
                    img_tensors.append(img_processed.detach().cpu())
                    crop_DL_pro = crop_model(img_processed).softmax(dim=-1).detach().cpu().numpy()
                    
                    target_df = test_dataset.df.iloc[BATCH_SIZE*img_idx: BATCH_SIZE*(img_idx + 1), :].drop(columns=['img_path'])
                    crop_boost_pro = crop_cat.predict_proba(target_df)
                    crop_idx_list = np.argmax(crop_DL_pro + crop_boost_pro, axis=-1).tolist()
                    preds_crop.extend(crop_idx_list)
                    
            elif current_label == 'disease':
                for img_idx, data in tqdm(enumerate(test_dataloader)):
                    disease_mask = data
                    img_processed = img_tensors[img_idx].to(DEVICE)
                    disease_mask_np = disease_mask.numpy()
                    disease_mask = disease_mask.to(DEVICE)
                    
                    disease_DL_pro = disease_model(img_processed, disease_mask).softmax(dim=-1).detach().cpu().numpy()
                    
                    target_df = test_dataset.df.iloc[BATCH_SIZE*img_idx: BATCH_SIZE*(img_idx + 1), :].drop(columns=['img_path'])
                    
                    disease_boost_pro = disease_cat.predict_proba(target_df)
                    disease_boost_pro = np.ma.MaskedArray(data=disease_boost_pro, mask=disease_mask_np)
                    disease_idx_list = np.argmax(disease_DL_pro + disease_boost_pro.data, axis=-1).tolist()
                    preds_disease.extend(disease_idx_list)
                    
            elif current_label == 'risk':
                for img_idx, data in tqdm(enumerate(test_dataloader)):
                    disease, risk_mask = data
                    img_processed = img_tensors[img_idx].to(DEVICE)
                    risk_mask_np = risk_mask.numpy()
                    disease = disease.to(DEVICE)
                    risk_mask = risk_mask.to(DEVICE)
                    
                    risk_idx_list = torch.argmax(risk_model(img_processed, disease, risk_mask), dim=-1).detach().cpu().tolist()
                    
                    preds_risk.extend(risk_idx_list)
                
                
preds_crop = [str(i+1) for i in preds_crop]
preds_disease = [disease_decoding[i] if i != 0 else '00' for i in preds_disease]
preds_risk = [str(i) for i in preds_risk]
img_paths = test_dataset.df['img_path'].to_list()

predicted_answer = [[re.findall(r'[0-9]+', image_path)[-1], i + '_' + j + '_' + k] for image_path, i, j, k in zip(img_paths, preds_crop, preds_disease, preds_risk)]
column_names = ['image', 'label']

with open(args.csv_save, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    writer.writerow(column_names)
    writer.writerows(predicted_answer)
