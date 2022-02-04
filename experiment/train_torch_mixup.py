import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from model import BaseModel, CropHeadClassifier, DiseaseHeadClassifier, RiskHeadClassifier
from utils_ver2_mixin import TrainValDataset, total_acc_cal, get_sampler_weight
#from utils_ver2_mixin import TrainValDataset
from sklearn.metrics import f1_score, accuracy_score
from meta_data import disease_encoding
from loss import FocalLoss
from catboost import CatBoostClassifier
import argparse
import os
import glob
import numpy as np

parser = argparse.ArgumentParser(description='Select Train Mode')
#parser.add_argument('--target', type=str, required=True, choices=['crop', 'disease', 'risk'], help='Select target label.')
parser.add_argument('--save-folder', type=str, default=None, help='Save folder for Model. Recommand Not To Use.')
parser.add_argument('--dataset-folder', type=str, default='strat_dataset_ver2', help='dataset folder made by data preprocessing')
parser.add_argument('--checkpoint_cat', type=str, default='model_checkpoint/boosting', help='Checkpoint file for pytorch model', required=False)

args = parser.parse_args()

BASE_SAVE_FOLDER = 'model_checkpoint/torch'
#MODELS = {'crop': CropClassifier, 'disease': DiseaseClassifier, 'risk': RiskClassifier}
BASE_DROP_RATE = 0.3 #Cait model don't use it.
CROP_DROP_RATE = 0.3
DISEASE_DROP_RATE = 0.3
RISK_DROP_RATE = 0.3
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.01 #default=1e-2, tried:0.015, 0
BATCH_SIZE = 44
SAMPLER_SIZE = 0.8 #total data in one epoch is BATCH_SIZE * SAMPLER_SIZE
NUM_WORKERS = 4
EARLY_STOP_COUNT = 250
MIXUP_ALPHA = 0.5
MIXUP_BETA = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU_DEVICE = torch.device('cpu')

DL_RATIO_CROP = 0.5 
CAT_RATIO_CROP = 1 - DL_RATIO_CROP
DL_RATIO_DISEASE = 0.5 
CAT_RATIO_DISEASE = 1 - DL_RATIO_DISEASE
DL_RATIO_RISK = 1
CAT_RATIO_RISK = 1 - DL_RATIO_RISK

#target_label = args.target
save_folder = args.save_folder
dataset_folder = args.dataset_folder
if save_folder:
    os.makedirs(save_folder, exist_ok=True)
else:
    #if target_label in ('crop', 'disease', 'risk'):
    save_folder = os.path.join(BASE_SAVE_FOLDER, 'one_model_ver2_4')
    os.makedirs(save_folder, exist_ok=True)

total_crop_labels = [i for i in range(6)]
total_disease_labels = [i for i in range(len(disease_encoding))]
total_risk_labels = [i for i in range(4)]
# answer_total_idx = []
# for i in range(6):
#     for j in range(len(disease_encoding)):
#         for k in range(4):
#             answer_total_idx.append(j*100 + i*10 + k)
# answer_total_idx.sort()
# answer_total_dict = {v: i for i, v in enumerate(answer_total_idx)}
# total_total_labels = [answer_total_dict[disease_idx*100 + crop_idx*10 + risk_idx] for crop_idx, disease_idx, risk_idx in zip(total_crop_labels, total_disease_labels, total_risk_labels)]

train_csv_files = sorted(glob.glob(dataset_folder + '/*train.csv'))
test_csv_files = sorted(glob.glob(dataset_folder + '/*test.csv'))

crop_cat = CatBoostClassifier()
disease_cat = CatBoostClassifier()
risk_cat = CatBoostClassifier()
for cat_file_path, cat_model in zip(sorted(glob.glob(args.checkpoint_cat+'/*/max_f1*')), [crop_cat, disease_cat, risk_cat]):
    cat_model.load_model(cat_file_path)


for k_fold, (train_csv, test_csv) in enumerate(zip(train_csv_files, test_csv_files)):
    print("Train csv and Test csv:", train_csv, '\t', test_csv)
    """
    연습. 하단걸 지우면 됨
    """
    #train_csv = 'processed_csv.csv'
    #test_csv = 'processed_csv.csv'
    train_dataset = TrainValDataset(train_csv, train=True, target_label='total', mixup=True, mixup_alpha=MIXUP_ALPHA, mixup_beta=MIXUP_BETA)
    test_dataset = TrainValDataset(test_csv, train=False, target_label='total')
    #sampler_weight = get_sampler_weight(train_dataset.df['strat'])
    #sampler = WeightedRandomSampler(sampler_weight, len(sampler_weight))
    #sampler = WeightedRandomSampler(sampler_weight, BATCH_SIZE*SAMPLER_SIZE, replacement=False)
    #sampler = WeightedRandomSampler(sampler_weight, int(len(train_dataset)*SAMPLER_SIZE), replacement=False)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=BATCH_SIZE*2, drop_last=True, pin_memory=True)
    #train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=BATCH_SIZE*2, drop_last=True, pin_memory=True)
    
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
    
    loss_fn = FocalLoss() #torch.nn.CrossEntropyLoss()
    loss_fn_disease_risk = FocalLoss(ignore_index=0) #torch.nn.CrossEntropyLoss(ignore_index=0)
    # optimizer = torch.optim.SGD([
    #     {'params': base_model.parameters()}, 
    #     {'params': crop_model.parameters()}, 
    #     {'params': disease_model.parameters()}, 
    #     {'params': risk_model.parameters()}], lr=LEARNING_RATE, momentum=0.9
    # )
    optimizer = torch.optim.AdamW([
        {'params': base_model.parameters()}, 
        {'params': crop_model.parameters()}, 
        {'params': disease_model.parameters()}, 
        {'params': risk_model.parameters()}], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    #optimizer = torch.optim.AdamW([base_model.parameters(), crop_model.parameters(), disease_model.parameters(), risk_model.parameters()], lr=LEARNING_RATE)
    writer = SummaryWriter(os.path.join(save_folder, 'log'))#, f'log_{k_fold}'))
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
    answer_crop = test_dataset.df1['crop'].to_list()
    answer_disease = test_dataset.df1['disease'].to_list()
    answer_risk = test_dataset.df1['risk'].to_list()
    
    #answer_total_idx = [disease_idx*100+crop_idx*10+risk_idx for crop_idx, disease_idx, risk_idx in zip(answer_crop, answer_disease, answer_risk)]
    #for i in range(6):
    #    for j in range(len(disease_encoding)):
    #        for k in range(4):
    #            answer_total_idx.append(j*100 + i*10 + k)
    #answer_total_idx = sorted(list(set(answer_total_idx)))
    #answer_total_dict = {v: i for i, v in enumerate(answer_total_idx)}
    #answer_total = [answer_total_dict[disease_idx*100 + crop_idx*10 + risk_idx] for crop_idx, disease_idx, risk_idx in zip(answer_crop, answer_disease, answer_risk)]
    answer_total = [disease_idx*100 + crop_idx*10 + risk_idx for crop_idx, disease_idx, risk_idx in zip(answer_crop, answer_disease, answer_risk)]

    
    while True:
        stop_count += 1
        current_epoch += 1
        base_model.train()
        crop_model.train()
        disease_model.train()
        risk_model.train()
        train_dataset._shuffle_dataframe()
        for data in train_dataloader:
            img, crop1, disease1, risk1, disease_mask1, risk_mask1, crop2, disease2, risk2, disease_mask2, risk_mask2, beta_value = data
            # img = img.to(DEVICE)
            # crop = crop.to(DEVICE)
            # disease = disease.to(DEVICE)
            # risk = risk.to(DEVICE)
            # disease_mask = disease_mask.to(DEVICE)
            # risk_mask = risk_mask.to(DEVICE)
            with torch.cuda.amp.autocast():
                img = img.to(DEVICE)
                crop1 = crop1.to(DEVICE)
                disease1 = disease1.to(DEVICE)
                risk1 = risk1.to(DEVICE)
                disease_mask1 = disease_mask1.to(DEVICE)
                risk_mask1 = risk_mask1.to(DEVICE)
                crop2 = crop2.to(DEVICE)
                disease2 = disease2.to(DEVICE)
                risk2 = risk2.to(DEVICE)
                disease_mask2 = disease_mask2.to(DEVICE)
                risk_mask2 = risk_mask2.to(DEVICE)
                beta_value = beta_value.to(DEVICE)
                img_processed = base_model(img)
                # From Here. I found batch size problem.
                crop_logits = crop_model(img_processed)
                disease_logits1 = disease_model(img_processed, disease_mask1)
                risk_logits1 = risk_model(img_processed, disease1, risk_mask1)
                loss1 = loss_fn(crop_logits, crop1, beta_value) + loss_fn_disease_risk(disease_logits1, disease1, beta_value) + loss_fn_disease_risk(risk_logits1, risk1, beta_value) #1.5, 15
                
                disease_logits2 = disease_model(img_processed, disease_mask2)
                risk_logits2 = risk_model(img_processed, disease2, risk_mask2)
                loss2 = loss_fn(crop_logits, crop2, 1- beta_value) + loss_fn_disease_risk(disease_logits2, disease2, 1- beta_value) + loss_fn_disease_risk(risk_logits2, risk2, 1- beta_value)
                
                loss = loss1 + loss2
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            #loss.backward()
            #optimizer.step()
            #score = accuracy_score(labels.cpu(), torch.argmax(logits, dim=-1).detach().cpu())
            writer.add_scalar(f'Loss/train_step_{k_fold}', loss, step)
            #writer.add_scalar(f'Acc/train_step_{k_fold}', score, step)
            step += 1
        train_dataset._shuffle_dataframe()
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
                
                    target_df = test_dataset.df1.iloc[BATCH_SIZE*idx: BATCH_SIZE*(idx + 1), :].drop(columns=['img_path', 'ordinate', 'crop', 'disease', 'risk'])#, 'strat'])
                    crop_boost_pro = crop_cat.predict_proba(target_df)
                    crop_DL_pro = crop_logits.softmax(dim=-1).detach().cpu().numpy()
                    crop_idx_list = np.argmax(crop_DL_pro*DL_RATIO_CROP + crop_boost_pro*CAT_RATIO_CROP, axis=-1).tolist()
                    preds_crop_ensem.extend(crop_idx_list)
                    
                    
                    disease_DL_pro = disease_logits.softmax(dim=-1).detach().cpu().numpy()
                    target_df = test_dataset.df1.iloc[BATCH_SIZE*idx: BATCH_SIZE*(idx + 1), :].drop(columns=['img_path', 'ordinate', 'disease', 'risk'])#, 'strat'])
                    disease_boost_pro = disease_cat.predict_proba(target_df)
                    disease_boost_pro = np.ma.MaskedArray(data=disease_boost_pro, mask=disease_mask_np)
                    disease_idx_list = np.argmax(disease_DL_pro*DL_RATIO_DISEASE + disease_boost_pro.data*CAT_RATIO_DISEASE, axis=-1).tolist()
                    preds_disease_ensem.extend(disease_idx_list)
                
                
                total_loss += loss * len(img)
                preds_crop.extend(np.argmax(crop_DL_pro, axis=-1))
                preds_disease.extend(np.argmax(disease_DL_pro, axis=-1))
                preds_risk.extend(torch.argmax(risk_logits, dim=-1).detach().cpu().tolist())
                # preds_crop.extend(torch.argmax(crop_logits, dim=-1).detach().cpu().tolist())
                # preds_disease.extend(torch.argmax(disease_logits, dim=-1).detach().cpu().tolist())
                # preds_risk.extend(torch.argmax(risk_logits, dim=-1).detach().cpu().tolist())
                
                
                
            preds_total = [disease_idx*100 + crop_idx*10 + risk_idx for crop_idx, disease_idx, risk_idx in zip(preds_crop, preds_disease, preds_risk)]
            preds_total_ensem = [disease_idx*100 + crop_idx*10 + risk_idx for crop_idx, disease_idx, risk_idx in zip(preds_crop_ensem, preds_disease_ensem, preds_risk)]
            f1_score_crop = f1_score(answer_crop, preds_crop, average='macro', labels=total_crop_labels)
            f1_score_crop_ensem = f1_score(answer_crop, preds_crop_ensem, average='macro', labels=total_crop_labels)
            f1_score_disease = f1_score(answer_disease, preds_disease, average='macro', labels=total_disease_labels)
            f1_score_disease_ensem = f1_score(answer_disease, preds_disease_ensem, average='macro', labels=total_disease_labels)
            f1_score_risk = f1_score(answer_risk, preds_risk, average='macro', labels=total_risk_labels)
            f1_score_total = f1_score(answer_total, preds_total, average='macro')#, labels=total_total_labels)
            f1_score_total_ensem = f1_score(answer_total, preds_total_ensem, average='macro')#, labels=total_total_labels)
            
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
        #     torch.save({'base_model': base_model.state_dict(),
        #                 'crop_model': crop_model.state_dict(),
        #                 'disease_model': disease_model.state_dict(),
        #                 'risk_model': risk_model.state_dict(),}, os.path.join(save_folder, f'max_f1_average_{k_fold}.pt'))
            
        if max_f1_total < f1_score_total:
            stop_count = 0
            max_f1_total = f1_score_total
            torch.save({'base_model': base_model.state_dict(),
                        'crop_model': crop_model.state_dict(),
                        'disease_model': disease_model.state_dict(),
                        'risk_model': risk_model.state_dict(),}, os.path.join(save_folder, f'max_f1_total_{k_fold}.pt'))
        
        if max_f1_ensemble < f1_score_total_ensem:
            stop_count = 0
            max_f1_ensemble = f1_score_total_ensem
            torch.save({'base_model': base_model.state_dict(),
                        'crop_model': crop_model.state_dict(),
                        'disease_model': disease_model.state_dict(),
                        'risk_model': risk_model.state_dict(),}, os.path.join(save_folder, f'max_f1_ensem_{k_fold}.pt'))
        
        if max_f1_total_risk_care < f1_score_total and f1_score_crop == 1 and f1_score_disease == 1:
            stop_count = 0
            max_f1_total_risk_care = f1_score_total
            print(f'DL only F1 Score: {max_f1_total_risk_care}')
            torch.save({'base_model': base_model.state_dict(),
                        'crop_model': crop_model.state_dict(),
                        'disease_model': disease_model.state_dict(),
                        'risk_model': risk_model.state_dict(),}, os.path.join(save_folder, f'max_f1_total_risk_care_{k_fold}.pt'))
        
        if max_f1_ensemble_risk_care < f1_score_total_ensem and f1_score_crop_ensem == 1 and f1_score_disease_ensem == 1:
            stop_count = 0
            max_f1_ensemble_risk_care = f1_score_total_ensem
            print(f'Ensemble F1 Score: {max_f1_ensemble_risk_care}')
            torch.save({'base_model': base_model.state_dict(),
                        'crop_model': crop_model.state_dict(),
                        'disease_model': disease_model.state_dict(),
                        'risk_model': risk_model.state_dict(),}, os.path.join(save_folder, f'max_f1_ensem_risk_care_{k_fold}.pt'))
        
        # if max_average_acc < acc_average_score:
        #     stop_count = 0
        #     max_average_acc = acc_average_score
        #     torch.save({'base_model': base_model.state_dict(),
        #                 'crop_model': crop_model.state_dict(),
        #                 'disease_model': disease_model.state_dict(),
        #                 'risk_model': risk_model.state_dict(),}, os.path.join(save_folder, f'max_average_acc_{k_fold}.pt'))
            
        # if max_all_acc < acc_all_score:
        #     stop_count = 0
        #     max_all_acc = acc_all_score
        #     torch.save({'base_model': base_model.state_dict(),
        #                 'crop_model': crop_model.state_dict(),
        #                 'disease_model': disease_model.state_dict(),
        #                 'risk_model': risk_model.state_dict(),}, os.path.join(save_folder, f'max_all_acc_{k_fold}.pt'))
        
        if stop_count == EARLY_STOP_COUNT:
            print(f'{k_fold}: Training Complete.')
            break
