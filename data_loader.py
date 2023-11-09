import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa


# 创建测试数据集
class MVTecUNETDRAEMTestDataset(Dataset):
    
    # 参数为测试数据集根目录
    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        # 原始图片路径
        self.images = sorted(glob.glob(root_dir+"/img/*.png"))
        self.recons_images = sorted(glob.glob(root_dir+"/img_recons/*.png"))
        self.mask_images = sorted(glob.glob(root_dir+"/mask/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, recons_path, mask_path):
        # 得到原始图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # 得到重建图像
        recons_image = cv2.imread(recons_path, cv2.IMREAD_COLOR)
        
        # 得到mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        recons_image = recons_image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        recons_image = np.array(recons_image).reshape((recons_image.shape[0], recons_image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        recons_image = np.transpose(recons_image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        
        return image, recons_image, mask

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # 得到原始图片路径
        img_path = self.images[idx]
        
        # 得到重建图像路径
        recons_path = self.recons_images[idx]
        
        # 得到mask路径
        mask_path = self.mask_images[idx]
        
        # 得到image和mask    
        image, recons_image, mask = self.transform_image(img_path, recons_path, mask_path)
        
        # 判断是否有异常或全0mask
        non_zero_pixels = np.any(mask != 0)
        
        # 如果有异常
        if non_zero_pixels:
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'recons_image': recons_image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}
        
        return sample


# 定义根据img, img_recons, mask三个文件夹得到数据的数据集类
class MVTecUNETDRAEMTrainDataset(Dataset):
    
    def __init__(self, root_dir, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.resize_shape=resize_shape
        
        # 原始图片路径
        self.images = sorted(glob.glob(root_dir+"/img/*.png"))
        self.recons_images = sorted(glob.glob(root_dir+"/img_recons/*.png"))
        self.mask_images = sorted(glob.glob(root_dir+"/mask/*.png"))
        
        # 设置系列数据增强
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]
        
        # 设置旋转
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.images)
    
    
    def transform_image(self, image_path, recons_path, mask_path):
        # 得到原始图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # 得到重建图像
        recons_image = cv2.imread(recons_path, cv2.IMREAD_COLOR)
        
        # 得到mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        recons_image = recons_image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        recons_image = np.array(recons_image).reshape((recons_image.shape[0], recons_image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        recons_image = np.transpose(recons_image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        
        return image, recons_image, mask
    
    def __getitem__(self, idx):
        # 获得idx
        idx = torch.randint(0, len(self.images), (1,)).item()
        # recons_idx = torch.randint(0, len(self.recons_images), (1,)).item()
        # mask_idx = torch.randint(0, len(self.mask_images), (1,)).item()
        recons_idx = idx
        mask_idx = idx
        
        image, recons_img, mask = self.transform_image(self.images[idx], self.recons_images[recons_idx], self.mask_images[mask_idx])
        
        sample = {'image': image, "recons_img": recons_img, 'mask': mask, 'idx': idx}
        
        return sample
