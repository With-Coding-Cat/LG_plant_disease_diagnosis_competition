import argparse
import glob
import re
import csv
from multiprocessing import Pool
from tqdm import tqdm
from catboost import CatBoostClassifier
import torch
from torch.utils.data import DataLoader
from model import BaseModel, CropHeadClassifier, DiseaseHeadClassifier, RiskHeadClassifier
from utils import TrainValDataset, ensemble_cal, ensemble_cal_with_mask
from meta_data import disease_encoding, disease_decoding

parser = argparse.ArgumentParser(description='It is for predict')
parser.add_argument('--checkpoint-torch', type=str, default='model_checkpoint/torch/trained/max_f1_total_0.pt', help='Checkpoint file for pytorch model', required=False)
parser.add_argument('--checkpoint-cat-crop', type=str, default='model_checkpoint/boosting/crop/crop_0.pkl', help='Checkpoint file for pytorch model', required=False)
parser.add_argument('--checkpoint-cat-disease', type=str, default='model_checkpoint/boosting/disease/disease_0.pkl', help='Checkpoint file for pytorch model', required=False)
parser.add_argument('--csv-file', type=str, default='preprocessed_test.csv', required=False)
parser.add_argument('--csv-save', type=str, default='result.csv', required=False)
parser.add_argument('--batch-size', type=int, default=180, required=False)

args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = args.batch_size
NUM_WORKERS = 4
total_crop_labels = [i for i in range(6)]
total_disease_labels = [i for i in range(len(disease_encoding))]
total_risk_labels = [i for i in range(4)]
test_csv = args.csv_file
target_labels = ['crop', 'disease', 'risk']

print("Start Predict")

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
crop_cat.load_model(args.checkpoint_cat_crop)
disease_cat.load_model(args.checkpoint_cat_disease)

base_model.eval()
crop_model.eval()
disease_model.eval()
risk_model.eval()

test_dataset = TrainValDataset(test_csv, train=False, target_label='crop')
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=BATCH_SIZE*2, drop_last=False, pin_memory=True)

img_tensors = []
preds_crop = []
preds_disease = []
preds_risk = []

for current_label in target_labels:
    if current_label != 'crop':
        if current_label == 'disease':
            test_dataset._change_label_and_update_crop(preds_crop)
        else:
            test_dataset._change_label_and_update_disease(preds_disease)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            if current_label == 'crop':
                list_for_crop_prediction = []
                for img_idx, data in tqdm(enumerate(test_dataloader)):
                    img = data
                    img = img.to(DEVICE)
                    img_processed = base_model(img)
                    img_tensors.append(img_processed.detach().cpu())
                    crop_DL_pro = crop_model(img_processed).softmax(dim=-1).detach().cpu().numpy()
                    target_df = test_dataset.df.iloc[BATCH_SIZE*img_idx: BATCH_SIZE*(img_idx + 1), :].drop(columns=['img_path'])
                    list_for_crop_prediction.append((crop_DL_pro, target_df, crop_cat))
                
                with Pool() as pool:
                    preds_crop = pool.map(ensemble_cal, list_for_crop_prediction)
                preds_crop = [index for index_list in preds_crop for index in index_list]
                    
            elif current_label == 'disease':
                list_for_disease_prediction  = []
                for img_idx, data in tqdm(enumerate(test_dataloader)):
                    disease_mask = data
                    img_processed = img_tensors[img_idx].to(DEVICE)
                    disease_DL_pro = disease_model(img_processed, disease_mask.to(DEVICE)).softmax(dim=-1).detach().cpu().numpy()
                    target_df = test_dataset.df.iloc[BATCH_SIZE*img_idx: BATCH_SIZE*(img_idx + 1), :].drop(columns=['img_path'])
                    list_for_disease_prediction.append((disease_DL_pro, target_df, disease_cat, disease_mask))
                
                with Pool() as pool:
                    preds_disease = pool.map(ensemble_cal_with_mask, list_for_disease_prediction)
                preds_disease = [index for index_list in preds_disease for index in index_list]
                    
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
