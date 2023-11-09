import os
import torch
from torchvision import transforms
import torch.nn.functional as F
from data_loader import MVTecUNETDRAEMTestDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from PIL import Image
import cv2


def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc,"+run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc,"+run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap,"+run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap,"+run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += ","+str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt",'a+') as file:
        file.write(fin_str)


# 测试数据集
def test(root_dir, checkpoint_path, base_model_name):
    
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    
    
    # 测试代码块
    img_dim = 256
    run_name = base_model_name+"_"+'_'

    # model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    # model.load_state_dict(torch.load(os.path.join(checkpoint_path,run_name+".pckl"), map_location='cuda:0'))
    # model.cuda()
    # model.eval()

    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.load_state_dict(torch.load(os.path.join(checkpoint_path, run_name+"_seg.pckl"), map_location='cuda:0'))
    model_seg.cuda()
    model_seg.eval()

    dataset = MVTecUNETDRAEMTestDataset(root_dir, resize_shape=[img_dim, img_dim])
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    total_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(dataset)))
    mask_cnt = 0

    anomaly_score_gt = []
    anomaly_score_prediction = []

    display_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    display_gt_images = torch.zeros((16 ,3 ,256 ,256)).cuda()
    display_out_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    display_in_masks = torch.zeros((16 ,1 ,256 ,256)).cuda()
    cnt_display = 0
    
    display_indices = np.random.randint(len(dataloader), size=(16,))
    
    # 创建可视化结果文件集
    os.makedirs("./display_images/", exist_ok=True)
    
    for i_batch, sample_batched in enumerate(dataloader):
        
        # 得到原始图片
        img = sample_batched["image"].cuda()
        
        # 得到重建图片
        gray_rec = sample_batched["recons_image"].cuda()
        
        # 是否为正常样本
        is_normal = sample_batched["has_anomaly"].detach().numpy()[0 ,0]
        anomaly_score_gt.append(is_normal)
        
        # 得到真实mask
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask.detach().numpy()[0, :, :, :].transpose((1, 2, 0))
        
        joined_in = torch.cat((gray_rec.detach(), img), dim=1)

        out_mask = model_seg(joined_in)

        out_mask_sm = torch.softmax(out_mask, dim=1)
        
        show_mask = out_mask_sm[:, 1:, :, :]
        show_mask = show_mask[0]
        
        # 将 PyTorch 张量转换为 NumPy 数组
        show_mask = show_mask.cpu()
        show_mask_np = show_mask.detach().numpy()
        show_mask_np = show_mask_np.squeeze()

        # 创建 PIL Image 对象
        show_mask_image = Image.fromarray((show_mask_np * 255).astype(np.uint8))

        # 保存为 jpg 格式图像
        show_mask_image.save("./display_images/show_mask_{}.jpg".format(i_batch))
        
        # --------------------------------------------------------------------
        
        if i_batch in display_indices:
            t_mask = out_mask_sm[:, 1:, :, :]
            display_images[cnt_display] = gray_rec[0]
            display_gt_images[cnt_display] = img[0]
            display_out_masks[cnt_display] = t_mask[0]
            display_in_masks[cnt_display] = true_mask[0]
            cnt_display += 1

        out_mask_cv = out_mask_sm[0 ,1 ,: ,:].detach().cpu().numpy()

        out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask_sm[: ,1: ,: ,:], 21, stride=1, padding=21 // 2).cpu().detach().numpy()
        
        image_score = np.max(out_mask_averaged)

        anomaly_score_prediction.append(image_score)

        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)
    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    obj_ap_pixel_list.append(ap_pixel)
    obj_auroc_pixel_list.append(auroc_pixel)
    obj_auroc_image_list.append(auroc)
    obj_ap_image_list.append(ap)
    
    # 计算指标
    print("AUC Image:  " +str(auroc))
    print("AP Image:  " +str(ap))
    print("AUC Pixel:  " +str(auroc_pixel))
    print("AP Pixel:  " +str(ap_pixel))
    print("==============================")
    print(run_name)
    write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)


if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', action='store', type=int, required=True)
    parser.add_argument('--root_dir', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    
    args = parser.parse_args()
    
    with torch.cuda.device(args.gpu_id):
        test(args.root_dir, args.checkpoint_path, args.base_model_name)
