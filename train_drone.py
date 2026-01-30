import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import shutil
from tqdm import tqdm
import json

# 引入你的模块
from model.net import HomoNet
from model.swin_multi import SwinTransformer
from mamba_nocam import MambaIC_NoCAM 
from model_drone import DroneVideoCompressor
from dataset_drone import DroneVideoDataset 


class Params:
    """Helper class that loads parameters from a json file."""
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, dict_vals):
        self.__dict__.update(dict_vals)

# --- A. 辅助函数：配置优化器 (参考 m_train.py) ---
def configure_optimizers(net, args):
    """
    分离网络参数和熵模型参数，使用不同的优化器
    """
    parameters = {
        n: p
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n: p
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # 1. 主优化器: 优化卷积、Mamba等权重 (参考 m_train.py 的设置)
    # 学习率通常设为 1e-4
    optimizer = optim.AdamW(
        (p for p in parameters.values()), 
        lr=args.lr, 
        weight_decay=1e-4
    )

    # 2. 辅助优化器: 专门优化熵模型的概率分布参数 (必须有！)
    # 学习率通常比主优化器大，约 1e-3
    aux_optimizer = optim.Adam(
        (p for p in aux_parameters.values()), 
        lr=args.aux_lr
    )

    return optimizer, aux_optimizer

# --- B. Loss 函数 ---
def rate_distortion_loss(output, target, lmbda=0.01):
    # 1. Distortion (MSE)
    mse_loss = nn.functional.mse_loss(output["rec_frame"], target)
    
    # 2. Rate (Bits)
    likelihoods = output["likelihoods"]
    N, _, H, W = target.shape
    num_pixels = N * H * W
    
    # 计算总比特数
    bpp_loss = sum(
        (torch.log(l).sum() / (-torch.log(torch.tensor(2.0)) * num_pixels))
        for l in likelihoods.values()
    )
    
    # Total Loss
    loss = lmbda * (255**2) * mse_loss + bpp_loss
    return loss, mse_loss, bpp_loss

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # # --- 1. 数据加载 ---
    # # [关键]: 尺寸必须与 HomoGAN 参数一致
    # train_patch_size = (640, 360) 
    
    # train_dataset = DroneVideoDataset(
    #     root_dir=args.data_dir,
    #     patch_size=train_patch_size,
    #     is_train=True
    # )

    # === [开始粘贴你的新代码] ===
    # ... 加载 Params ...
    json_path = os.path.join(os.path.dirname(__file__), 'params.json')
    # 确保前面已经定义了 class Params
    if not os.path.exists(json_path):
        print(f"Error: 找不到配置文件 {json_path}")
        return
        
    params = Params(json_path)

    # [关键]: 从 params.json 读取 crop_size 作为 dataset 的尺寸
    if isinstance(params.crop_size, list):
        train_patch_size = tuple(params.crop_size) 
    else:
        train_patch_size = params.crop_size

    # 打印确认一下
    print(f"Training Patch Size (H, W): {train_patch_size}")
    print(f"HomoGAN Input Channels: {params.in_channels}")

    train_dataset = DroneVideoDataset(
        root_dir=args.data_dir,
        patch_size=train_patch_size, 
        is_train=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print(f"Data Loaded: {len(train_dataset)} pairs.")

    # --- 2. 模型初始化 ---
    # class HomoArgs:
    #     filter_size = 15
    #     num_basis = 8
    #     # [修改 1]: 补全 input channels
    #     # HomoGAN 处理的是灰度图，所以通道数是 1
    #     in_channels = 1 
        
    #     # [修改 2]: 补全 crop_size
    #     # net.py 里使用的是 params.crop_size，不是 patch_size
    #     # 必须确保它和 dataset 的裁剪尺寸一致: [Height, Width]
    #     crop_size = [360, 640] 
        
    #     # 保留 patch_size 以防万一 (虽然 net.py 主要用 crop_size)
    #     patch_size = [360, 640]
      
    # homo_args = HomoArgs()
    # 2. [核心修改] 显式传入 backbone 参数，并且必须传类名 SwinTransformer
    # 注意：不要加括号 SwinTransformer()，是传类本身
    homo_net = HomoNet(params, backbone=SwinTransformer).to(device)
    
    # 加载 HomoGAN 权重
    # print(f"Loading HomoGAN: {args.homogan_path}")
    # ckpt = torch.load(args.homogan_path, map_location='cpu')
    # state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    # new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # homo_net.load_state_dict(new_state_dict, strict=False)


    # --- 加载 HomoGAN 权重 (安全过滤版) ---
    print(f"Loading HomoGAN: {args.homogan_path}")
    ckpt = torch.load(args.homogan_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        
        # [关键]: 跳过尺寸不匹配的 basis 和 mask_pred
        if 'basis' in name or 'mask_pred' in name:
            continue
            
        new_state_dict[name] = v
        
    # strict=False 允许我们漏掉 basis 不加载
    msg = homo_net.load_state_dict(new_state_dict, strict=False)
    print(f"权重加载完毕。忽略了不匹配的层: {len(msg.missing_keys)} 个 (这是正常的)")

    # MambaIC
    mamba_net = MambaIC_NoCAM(N=128, M=192).to(device)
    
    # 组装
    model = DroneVideoCompressor(homo_net, mamba_net).to(device)
    
    # --- 3. 配置优化器 (关键修改) ---
    # 冻结 HomoGAN
    for param in model.homogan.parameters():
        param.requires_grad = False
        
    optimizer, aux_optimizer = configure_optimizers(model, args)
    
    # --- 4. 训练循环 ---
    global_step = 0
    best_loss = float('inf')
    
    model.train()
    for epoch in range(args.epochs):
        print(f"=== Epoch {epoch+1}/{args.epochs} ===")
        progress_bar = tqdm(train_loader)
        
        for curr_frame, ref_frame in progress_bar:
            curr_frame = curr_frame.to(device)
            ref_frame = ref_frame.to(device)
            
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            
            # Forward
            out = model(curr_frame, ref_frame)
            
            # Loss
            loss, mse, bpp = rate_distortion_loss(out, curr_frame, lmbda=args.lmbda)
            
            # Backward
            loss.backward()
            
            # [关键]: 梯度裁剪 (参考 m_train.py)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            
            optimizer.step()
            
            # [关键]: 更新辅助优化器 (Entropy Parameters)
            # 必须调用 mamba_codec.aux_loss()，因为 aux_loss 定义在 CompressionModel 基类里
            # 但 DroneVideoCompressor 没有 aux_loss，所以要通过 .mamba_codec 访问
            aux_loss = model.mamba_codec.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            
            # Log
            progress_bar.set_description(
                f"L:{loss.item():.4f} | M:{mse.item():.5f} | B:{bpp.item():.3f}"
            )
            global_step += 1
        
        # 保存 checkpoint
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, "best_drone_model.pth")
            print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                        default='/root/HomoGAN-main/dataset/sequences/Train')
    
    parser.add_argument('--homogan_path', type=str, 
                        default='/root/HomoGAN-main/experiments/fine_tuning/fine_tuning.pth')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--aux_lr', type=float, default=1e-3, help="辅助优化器学习率")
    parser.add_argument('--lmbda', type=float, default=0.01, help="R-D Loss 平衡系数")
    parser.add_argument('--clip_max_norm', type=float, default=1.0, help="梯度裁剪阈值")
    
    args = parser.parse_args()
    train(args)