import torch
# 导入训练集
from data_loader import MVTecUNETDRAEMTrainDataset
from torch.utils.data import DataLoader
from torch import optim
from model_unet import ReconstructiveSubNetwork, DiscriminativeSubNetwork
from loss import FocalLoss, SSIM
import os
from tensorboard_visualizer import TensorboardVisualizer


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 初始化模型权重
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train_on_device(args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    
    # 准备训练
    # 设置训练实验名称
    run_name = 'DRAEM_UNET_train_'+str(args.lr)+'_'+str(args.epochs)+'_bs'+str(args.bs)+"_"+"_"

    visualizer = TensorboardVisualizer(log_dir=os.path.join(args.log_path, run_name+"/"))

    # model = ReconstructiveSubNetwork(in_channels=3, out_channels=3)
    # model.cuda()
    # model.apply(weights_init)
    
    # 创建分割模型
    model_seg = DiscriminativeSubNetwork(in_channels=6, out_channels=2)
    model_seg.cuda()
    model_seg.apply(weights_init)
    
    # 设置优化器
    optimizer = torch.optim.Adam([{"params": model_seg.parameters(), "lr": args.lr}])
    
    # 设置学习率下降计划
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8, args.epochs*0.9],gamma=0.2, last_epoch=-1)
    
    # 定义函数
    # loss_l2 = torch.nn.modules.loss.MSELoss()
    # loss_ssim = SSIM()
    loss_focal = FocalLoss()
    
    # 得到数据集
    # 传入四个路径（总体路径，原图片路径，重建之后的路径，GT路径）
    dataset = MVTecUNETDRAEMTrainDataset(args.root_dir, resize_shape=[256, 256])
    
    # 得到数据加载器
    dataloader = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=16)
    
    n_iter = 0
    
    for epoch in range(args.epochs):
        
        print("Epoch: "+str(epoch))
        
        for i_batch, sample_batched in enumerate(dataloader):
            
            # 这个就是数据增强之后的图片（对应img）
            img = sample_batched["image"].cuda()
            
            # 这个就是重建图片（对应img_recons）
            recons_img = sample_batched["recons_img"].cuda()
            
            # 这个就是GT（对应mask）
            mask = sample_batched["mask"].cuda()
            
            # 进行拼接
            joined_in = torch.cat((recons_img, img), dim=1)
            
            # 得到UNET分割模型输出结果
            out_mask = model_seg(joined_in)
            
            # 这个就是UNET得到的分割结果
            out_mask_sm = torch.softmax(out_mask, dim=1)
            
            # 计算分割损失
            segment_loss = loss_focal(out_mask_sm, mask)
            
            # 只计算一个UNET分割损失
            loss = segment_loss
            
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            print("{} segment_loss: ".format(i_batch), segment_loss)
            
            if args.visualize and n_iter % 200 == 0:
                visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
                
            if args.visualize and n_iter % 400 == 0:
                t_mask = out_mask_sm[:, 1:, :, :]
                visualizer.visualize_image_batch(img, n_iter, image_name='batch_img')
                visualizer.visualize_image_batch(recons_img, n_iter, image_name='batch_recons_img')
                visualizer.visualize_image_batch(mask, n_iter, image_name='batch_mask')
                visualizer.visualize_image_batch(t_mask, n_iter, image_name='mask_out')

            n_iter +=1
        
        scheduler.step()
        
        print("--------------------n_iter: {}----------------".format(n_iter))
        
        # torch.save(model.state_dict(), os.path.join(args.checkpoint_path, run_name+".pckl"))
        torch.save(model_seg.state_dict(), os.path.join(args.checkpoint_path, run_name+"_seg.pckl"))

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--root_dir', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        train_on_device(args)

