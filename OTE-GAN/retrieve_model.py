from tqdm import tqdm
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader

from model.model_LC import _NetG
from dataloader.infer_loader import RetinalDataset
from utils import *

if __name__ == '__main__':
    gpu_id = "0"
    device = torch.device("cuda:" + gpu_id)
    #dataset = "idrid"
    #dataset = "EyeQ"
    dataset = "IDRID"
    #dataset = "Drive"

    batch_size = 40
    #dataroot_tr = f"dataset/{dataset}_degraded"
    #dataroot_tr = f"dataset/{dataset}_final"
    #dataroot_ts = f"dataset/{dataset}_degraded/test"
    dataroot_tr = f"dataset/{dataset}"
    #folders = {'deg': 'deg', 'pre':'pre', 'mask':'mask'}
    folders = {'deg': 'deg', 'pre':'images', 'mask':'mask'}
    dataset_tr = RetinalDataset(dataroot_tr, folders)
    # dataset_ts = RetinalDataset(dataroot_ts, folders)
    dataloader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=False)
    # dataloader_ts = DataLoader(dataset_ts, batch_size=batch_size, shuffle=False)

    model = _NetG()
    model = nn.DataParallel(model).to(device)

    PSNRs = []
    SSIMs = []
    weight_ids = []
    weights_files = subfiles("SottGan/Experiment/exp11/checkpoint")
    for weight_file in tqdm(weights_files):
        wieght_id = weight_file.split("/")[-1].replace(".pth", "")
        weight_ids.append(wieght_id)
        checkpoint = torch.load(weight_file)['model']
        model.load_state_dict(checkpoint.state_dict())
        model.eval()
        
        PSNR = 0.
        SSIM = 0.
        with torch.no_grad():
            for degraded, ground_truth, mask, image_id, subject_id in dataloader_tr:
                bsz = degraded.size(0)
                enhanced = model(degraded.to(device))
                degraded_np = degraded.detach().clone().cpu().numpy().transpose(0, 2, 3, 1)
                enhanced_np = enhanced.detach().clone().cpu().numpy().transpose(0, 2, 3, 1)
                ground_truth_np = ground_truth.detach().clone().cpu().numpy().transpose(0, 2, 3, 1)
                mask_np = mask.detach().clone().cpu().numpy().transpose(0, 2, 3, 1)
                masked_enhanced_np = enhanced_np * mask_np
                for i in range(bsz):
                    degraded_np_case = degraded_np[i, ...]
                    enhanced_np_case = enhanced_np[i, ...]
                    masked_enhanced_np_case = masked_enhanced_np[i, ...]
                    ground_truth_np_case = ground_truth_np[i, ...]
      
                    PSNR += evaluatePSNR(masked_enhanced_np_case, ground_truth_np_case) 
                    SSIM += evaluateSSIM(masked_enhanced_np_case, ground_truth_np_case)
            
            # for degraded, ground_truth, mask, image_id, subject_id in dataloader_ts:
            #     bsz = degraded.size(0)
            #     enhanced = model(degraded.to(device))
            #     degraded_np = degraded.detach().clone().cpu().numpy().transpose(0, 2, 3, 1)
            #     enhanced_np = enhanced.detach().clone().cpu().numpy().transpose(0, 2, 3, 1)
            #     ground_truth_np = ground_truth.detach().clone().cpu().numpy().transpose(0, 2, 3, 1)
            #     mask_np = mask.detach().clone().cpu().numpy().transpose(0, 2, 3, 1)
            #     masked_enhanced_np = enhanced_np * mask_np
            #     for i in range(bsz):
            #         degraded_np_case = degraded_np[i, ...]
            #         enhanced_np_case = enhanced_np[i, ...]
            #         masked_enhanced_np_case = masked_enhanced_np[i, ...]
            #         ground_truth_np_case = ground_truth_np[i, ...]
      
            #         PSNR += evaluatePSNR(masked_enhanced_np_case, ground_truth_np_case)
            #         SSIM += evaluateSSIM(masked_enhanced_np_case, ground_truth_np_case)

        psnr_mean = PSNR / (len(dataset_tr))
        ssim_mean = SSIM / (len(dataset_tr))

        PSNRs.append(psnr_mean)
        SSIMs.append(ssim_mean)

    weight_ids_np = np.array(weight_ids)

    weight_best_psnr_idx = np.argmax(PSNRs)
    weight_ids.append(f"MAX PSNR={weight_ids_np[weight_best_psnr_idx]}")
    PSNRs.append(PSNRs[weight_best_psnr_idx])
    SSIMs.append(SSIMs[weight_best_psnr_idx])

    weight_best_ssim_idx = np.argmax(SSIMs)
    weight_ids.append(f"MAX SSIM={weight_ids_np[weight_best_ssim_idx]}")
    PSNRs.append(PSNRs[weight_best_ssim_idx])
    SSIMs.append(SSIMs[weight_best_ssim_idx])

    df_res = np.concatenate([np.array(weight_ids).reshape(-1, 1), 
                             np.array(PSNRs).reshape(-1, 1), 
                             np.array(SSIMs).reshape(-1, 1)], axis=-1)

    pd.DataFrame(df_res, columns=["weight", "PSNR", "SSIM"]).to_csv(f"{dataset}.csv", index=False)
