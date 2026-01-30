import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DroneVideoDataset(Dataset):
    def __init__(self, root_dir, patch_size=(384, 640), is_train=True):
        """
        patch_size: 最终输入给网络的尺寸 (Height, Width)，必须是 384x640
        """
        self.root_dir = root_dir
        self.target_height = patch_size[0] # 384
        self.target_width = patch_size[1]  # 640
        self.is_train = is_train
        self.samples = [] 
        
        # 1. 递归扫描数据
        print(f"正在扫描数据: {root_dir} ...")
        seq_dirs = []
        for root, dirs, files in os.walk(root_dir):
            png_files = glob.glob(os.path.join(root, "*.png"))
            if len(png_files) >= 2:
                seq_dirs.append(root)
        seq_dirs = sorted(seq_dirs)
        
        # 2. 构建帧对
        for seq_path in seq_dirs:
            frames = sorted(glob.glob(os.path.join(seq_path, "*.png")))
            for i in range(len(frames) - 1):
                self.samples.append((frames[i], frames[i+1]))

        print(f"Dataset loaded: {len(self.samples)} pairs.")

        # --- [核心修改] 保持比例缩放 + 随机裁剪 ---
        # 逻辑：原图 360x640 -> Resize到高度384 (宽度会自动变为约682) -> 裁剪出 384x640
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                # 1. 传入一个整数，表示将最小边(360)缩放到 384，长边(640)按比例放大
                transforms.Resize(self.target_height, antialias=True), 
                # 2. 在放大后的图里随机切一刀
                transforms.RandomCrop((self.target_height, self.target_width)), 
                # 3. 随机翻转增强
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5)
            ])
        else:
            # 测试集使用 CenterCrop (中心裁剪)，保证结果可复现
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(self.target_height, antialias=True),
                transforms.CenterCrop((self.target_height, self.target_width))
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ref_path, curr_path = self.samples[idx]
        
        ref_img = Image.open(ref_path).convert('RGB')
        curr_img = Image.open(curr_path).convert('RGB')
        
        ref_np = np.asarray(ref_img)
        curr_np = np.asarray(curr_img)
        
        # 为了保证两张图裁剪的位置完全一样，我们先拼起来再裁
        combined_np = np.concatenate([ref_np, curr_np], axis=-1)
        
        # 执行变换 (Resize -> Crop)
        combined_tensor = self.transform(combined_np)
        
        # 裁完后再拆开
        ref_tensor, curr_tensor = torch.chunk(combined_tensor, chunks=2, dim=0)
        
        return curr_tensor, ref_tensor