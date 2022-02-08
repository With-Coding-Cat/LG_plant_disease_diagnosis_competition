import argparse
import os
import glob
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, accuracy_score
from model import BaseModel, CropHeadClassifier, DiseaseHeadClassifier, RiskHeadClassifier
from utils import TrainValDataset, total_acc_cal
from meta_data import disease_encoding
from loss import FocalLoss

parser = argparse.ArgumentParser(description='Select Train Mode')
parser.add_argument('--save-folder', type=str, default=None, help='Save folder for Model. Recommand Not To Use.')
parser.add_argument('--dataset-folder', type=str, default='stratified_dataset', help='dataset folder made by data preprocessing')
parser.add_argument('--checkpoint_cat', type=str, default='model_checkpoint/boosting', help='Checkpoint file for pytorch model', required=False)
parser.add_argument('--base-drop', type=float, default=0.3)
parser.add_argument('--crop-drop', type=float, default=0.3)
parser.add_argument('--disease-drop', type=float, default=0.3)
parser.add_argument('--risk-drop', type=float, default=0.3)
parser.add_argument('--learning-rate', type=float, default=0.00001)
parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=44)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--early-stop', type=int, default=250)

args = parser.parse_args()

BASE_SAVE_FOLDER = 'model_checkpoint/torch'
BASE_DROP_RATE = args.base_drop
CROP_DROP_RATE = args.crop_drop
DISEASE_DROP_RATE = args.disease_drop
RISK_DROP_RATE = args.risk_drop
LEARNING_RATE = args.learning_rate
WEIGHT_DECAY = args.weight_decay
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
EARLY_STOP_COUNT = args.early_stop
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_folder = args.save_folder
dataset_folder = args.dataset_folder
if save_folder:
    os.makedirs(save_folder, exist_ok=True)
else:
    save_folder = os.path.join(BASE_SAVE_FOLDER, 'trained')
    os.makedirs(save_folder, exist_ok=True)

total_crop_labels = [i for i in range(6)]
total_disease_labels = [i for i in range(len(disease_encoding))]
total_risk_labels = [i for i in range(4)]

train_csv_files = sorted(glob.glob(dataset_folder + '/*train.csv'))
test_csv_files = sorted(glob.glob(dataset_folder + '/*test.csv'))

crop_cat = CatBoostClassifier()
disease_cat = CatBoostClassifier()
for cat_file_path, cat_model in zip(sorted(glob.glob(args.checkpoint_cat+'/*/*_0.pkl')), [crop_cat, disease_cat]):
    cat_model.load_model(cat_file_path)

for k_fold, (train_csv, test_csv) in enumerate(zip(train_csv_files, test_csv_files)):
    print("Train csv and Test csv:", train_csv, '\t', test_csv)
    train_dataset = TrainValDataset(train_csv, train=True, target_label='total')
    test_dataset = TrainValDataset(test_csv, train=False, target_label='total')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=BATCH_SIZE*2, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, prefetch_factor=BATCH_SIZE*2, drop_last=False, pin_memory=True)
    
    base_model = BaseModel(drop_p=BASE_DROP_RATE)
    base_model = torch.nn.DataParallel(base_model).to(DEVICE)
    crop_model = CropHeadClassifier(drop_p=CROP_DROP_RATE)
    crop_model = torch.nn.DataParallel(crop_model).to(DEVICE)
    disease_model = DiseaseHeadClassifier(drop_p=DISEASE_DROP_RATE)
    disease_model = torch.nn.DataParallel(disease_model).to(DEVICE)
    risk_model = RiskHeadClassifier(drop_p=RISK_DROP_RATE)
    risk_model = torch.nn.DataParallel(risk_model).to(DEVICE)
    
    scaler = torch.cuda.amp.GradScaler()
    
    loss_fn = FocalLoss()
    loss_fn_disease_risk = FocalLoss(ignore_index=0)
    optimizer = torch.optim.AdamW([
        {'params': base_model.parameters()}, 
        {'params': crop_model.parameters()}, 
        {'params': disease_model.parameters()}, 
        {'params': risk_model.parameters()}], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    writer = SummaryWriter(os.path.join(save_folder, 'log'))
    step = 0
    stop_count = 0
    current_epoch = 0
    max_f1_average = -float('inf')
    max_f1_total = -float('inf')
    max_f1_ensemble = -float('inf')
    max_f1_total_risk_care = -float('inf')
    max_f1_ensemble_risk_care = -float('inf')
    max_average_acc = -float('inf')
    max_all_acc = -float('inf')
    answer_crop = test_dataset.df['crop'].to_list()
    answer_disease = test_dataset.df['disease'].to_list()
    answer_risk = test_dataset.df['risk'].to_list()
    answer_total = [disease_idx*100 + crop_idx*10 + risk_idx for crop_idx, disease_idx, risk_idx in zip(answer_crop, answer_disease, answer_risk)]

    
    while True:
        stop_count += 1
        current_epoch += 1
        base_model.train()
        crop_model.train()
        disease_model.train()
        risk_model.train()
        
        for data in train_dataloader:
            img, crop, disease, risk, disease_mask, risk_mask = data
            
            with torch.cuda.amp.autocast():
                img = img.to(DEVICE)
                crop = crop.to(DEVICE)
                disease = disease.to(DEVICE)
                risk = risk.to(DEVICE)
                disease_mask = disease_mask.to(DEVICE)
                risk_mask = risk_mask.to(DEVICE)
                img_processed = base_model(img)
                crop_logits = crop_model(img_processed)
                disease_logits = disease_model(img_processed, disease_mask)
                risk_logits = risk_model(img_processed, disease, risk_mask)
                loss = loss_fn(crop_logits, crop) + loss_fn_disease_risk(disease_logits, disease) + loss_fn_disease_risk(risk_logits, risk) #1.5, 15
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            writer.add_scalar(f'Loss/train_step_{k_fold}', loss, step)
            step += 1
        
        base_model.eval()
        crop_model.eval()
        disease_model.eval()
        risk_model.eval()
        total_loss = 0
        preds_crop = []
        preds_disease = []
        preds_risk = []
        preds_crop_ensem = []
        preds_disease_ensem = []
        preds_risk_ensem = []
        with torch.no_grad():
            for idx, data in enumerate(test_dataloader):
                img, crop, disease, risk, disease_mask, risk_mask = data
                img = img.to(DEVICE)
                crop = crop.to(DEVICE)
                disease = disease.to(DEVICE)
                risk = risk.to(DEVICE)
                disease_mask_np = disease_mask.numpy()
                disease_mask = disease_mask.to(DEVICE)
                risk_mask = risk_mask.to(DEVICE)
                with torch.cuda.amp.autocast():
                    img_processed = base_model(img)
                    crop_logits = crop_model(img_processed)
                    disease_logits = disease_model(img_processed, disease_mask)
                    risk_logits = risk_model(img_processed, disease, risk_mask)
                    
                    loss = loss_fn(crop_logits, crop) + loss_fn_disease_risk(disease_logits, disease) + loss_fn_disease_risk(risk_logits, risk)
                
                    target_df = test_dataset.df.iloc[BATCH_SIZE*idx: BATCH_SIZE*(idx + 1), :].drop(columns=['img_path', 'coordinate', 'crop', 'disease', 'risk'])
                    crop_boost_pro = crop_cat.predict_proba(target_df)
                    crop_DL_pro = crop_logits.softmax(dim=-1).detach().cpu().numpy()
                    crop_idx_list = np.argmax(crop_DL_pro + crop_boost_pro, axis=-1).tolist()
                    preds_crop_ensem.extend(crop_idx_list)
                    
                    
                    disease_DL_pro = disease_logits.softmax(dim=-1).detach().cpu().numpy()
                    target_df = test_dataset.df.iloc[BATCH_SIZE*idx: BATCH_SIZE*(idx + 1), :].drop(columns=['img_path', 'coordinate', 'disease', 'risk'])
                    disease_boost_pro = disease_cat.predict_proba(target_df)
                    disease_boost_pro = np.ma.MaskedArray(data=disease_boost_pro, mask=disease_mask_np)
                    disease_idx_list = np.argmax(disease_DL_pro + disease_boost_pro.data, axis=-1).tolist()
                    preds_disease_ensem.extend(disease_idx_list)
                
                
                total_loss += loss * len(img)
                preds_crop.extend(np.argmax(crop_DL_pro, axis=-1))
                preds_disease.extend(np.argmax(disease_DL_pro, axis=-1))
                preds_risk.extend(torch.argmax(risk_logits, dim=-1).detach().cpu().tolist())
                
                
            preds_total = [disease_idx*100 + crop_idx*10 + risk_idx for crop_idx, disease_idx, risk_idx in zip(preds_crop, preds_disease, preds_risk)]
            preds_total_ensem = [disease_idx*100 + crop_idx*10 + risk_idx for crop_idx, disease_idx, risk_idx in zip(preds_crop_ensem, preds_disease_ensem, preds_risk)]
            f1_score_crop = f1_score(answer_crop, preds_crop, average='macro', labels=total_crop_labels)
            f1_score_crop_ensem = f1_score(answer_crop, preds_crop_ensem, average='macro', labels=total_crop_labels)
            f1_score_disease = f1_score(answer_disease, preds_disease, average='macro', labels=total_disease_labels)
            f1_score_disease_ensem = f1_score(answer_disease, preds_disease_ensem, average='macro', labels=total_disease_labels)
            f1_score_risk = f1_score(answer_risk, preds_risk, average='macro', labels=total_risk_labels)
            f1_score_total = f1_score(answer_total, preds_total, average='macro')
            f1_score_total_ensem = f1_score(answer_total, preds_total_ensem, average='macro')
            
            acc_score_crop = accuracy_score(answer_crop, preds_crop)
            acc_score_disease = accuracy_score(answer_disease, preds_disease)
            acc_score_risk = accuracy_score(answer_risk, preds_risk)
            #acc_score_total = accuracy_score(answer_total, preds_total)
            
            writer.add_scalar(f'Loss/test_epoch_{k_fold}', total_loss/len(test_dataset), current_epoch)
            writer.add_scalar(f'Acc/Crop_{k_fold}', acc_score_crop, current_epoch)
            writer.add_scalar(f'Acc/Disease_{k_fold}', acc_score_disease, current_epoch)
            writer.add_scalar(f'Acc/Risk_{k_fold}', acc_score_risk, current_epoch)
            #writer.add_scalar(f'Acc_total/test_epoch_{k_fold}', acc_score_total, current_epoch)
            writer.add_scalar(f'F1/Crop_{k_fold}', f1_score_crop, current_epoch)
            writer.add_scalar(f'F1/Disease_{k_fold}', f1_score_disease, current_epoch)
            writer.add_scalar(f'F1/Risk_{k_fold}', f1_score_risk, current_epoch)
            writer.add_scalar(f'F1/Total_{k_fold}', f1_score_total, current_epoch)
            writer.add_scalar(f'F1/En_Crop_{k_fold}', f1_score_crop_ensem, current_epoch)
            writer.add_scalar(f'F1/En_Disease_{k_fold}', f1_score_disease_ensem, current_epoch)
            writer.add_scalar(f'F1/En_Risk_{k_fold}', f1_score_risk, current_epoch)
            writer.add_scalar(f'F1/En_Total_{k_fold}', f1_score_total_ensem, current_epoch)
            
            f1_average_score = (f1_score_crop + f1_score_disease + f1_score_risk) / 3
            acc_average_score = (acc_score_crop + acc_score_disease + acc_score_risk) / 3
            acc_all_score = total_acc_cal(preds_crop, answer_crop, preds_disease, answer_disease, preds_risk, answer_risk)
            
            writer.add_scalar(f'F1/Average_{k_fold}', f1_average_score, current_epoch)
            writer.add_scalar(f'Acc/Average_{k_fold}', acc_average_score, current_epoch)
            writer.add_scalar(f'Acc/All_{k_fold}', acc_all_score, current_epoch)
        
        # if max_f1_average < f1_average_score:
        #     stop_count = 0
        #     max_f1_average = f1_average_score
        #     torch.save({'base_model': base_model.module.state_dict(),
        #                 'crop_model': crop_model.module.state_dict(),
        #                 'disease_model': disease_model.module.state_dict(),
        #                 'risk_model': risk_model.module.state_dict(),}, os.path.join(save_folder, f'max_f1_average_{k_fold}.pt'))
            
        if max_f1_total < f1_score_total:
            stop_count = 0
            max_f1_total = f1_score_total
            torch.save({'base_model': base_model.module.state_dict(),
                        'crop_model': crop_model.module.state_dict(),
                        'disease_model': disease_model.module.state_dict(),
                        'risk_model': risk_model.module.state_dict(),}, os.path.join(save_folder, f'max_f1_total_{k_fold}.pt'))
        
        if max_f1_ensemble < f1_score_total_ensem:
            stop_count = 0
            max_f1_ensemble = f1_score_total_ensem
            torch.save({'base_model': base_model.module.state_dict(),
                        'crop_model': crop_model.module.state_dict(),
                        'disease_model': disease_model.module.state_dict(),
                        'risk_model': risk_model.module.state_dict(),}, os.path.join(save_folder, f'max_f1_ensem_{k_fold}.pt'))
        
        if max_f1_total_risk_care < f1_score_total and f1_score_crop == 1 and f1_score_disease == 1:
            stop_count = 0
            max_f1_total_risk_care = f1_score_total
            print(f'DL only F1 Score: {max_f1_total_risk_care}')
            torch.save({'base_model': base_model.module.state_dict(),
                        'crop_model': crop_model.module.state_dict(),
                        'disease_model': disease_model.module.state_dict(),
                        'risk_model': risk_model.module.state_dict(),}, os.path.join(save_folder, f'max_f1_total_risk_care_{k_fold}.pt'))
        
        if max_f1_ensemble_risk_care < f1_score_total_ensem and f1_score_crop_ensem == 1 and f1_score_disease_ensem == 1:
            stop_count = 0
            max_f1_ensemble_risk_care = f1_score_total_ensem
            print(f'Ensemble F1 Score: {max_f1_ensemble_risk_care}')
            torch.save({'base_model': base_model.module.state_dict(),
                        'crop_model': crop_model.module.state_dict(),
                        'disease_model': disease_model.module.state_dict(),
                        'risk_model': risk_model.module.state_dict(),}, os.path.join(save_folder, f'max_f1_ensem_risk_care_{k_fold}.pt'))
        
        # if max_average_acc < acc_average_score:
        #     stop_count = 0
        #     max_average_acc = acc_average_score
        #     torch.save({'base_model': base_model.module.state_dict(),
        #                 'crop_model': crop_model.module.state_dict(),
        #                 'disease_model': disease_model.module.state_dict(),
        #                 'risk_model': risk_model.module.state_dict(),}, os.path.join(save_folder, f'max_average_acc_{k_fold}.pt'))
            
        # if max_all_acc < acc_all_score:
        #     stop_count = 0
        #     max_all_acc = acc_all_score
        #     torch.save({'base_model': base_model.module.state_dict(),
        #                 'crop_model': crop_model.module.state_dict(),
        #                 'disease_model': disease_model.module.state_dict(),
        #                 'risk_model': risk_model.module.state_dict(),}, os.path.join(save_folder, f'max_all_acc_{k_fold}.pt'))
        
        if stop_count == EARLY_STOP_COUNT:
            print(f'{k_fold}: Training Complete.')
            break