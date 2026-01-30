import torch
import torch.nn as nn
from model.utils import get_warp_flow

class DroneVideoCompressor(nn.Module):
    def __init__(self, homogan_model, mambaic_model):
        super().__init__()
        self.homogan = homogan_model
        self.mamba_codec = mambaic_model
        
        # 冻结 HomoGAN
        for param in self.homogan.parameters():
            param.requires_grad = False

    def rgb_to_gray(self, x):
        return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    def get_homogan_flow(self, ref_gray, curr_gray):
        # 1. 提取特征
        ref_fea = self.homogan.fea_extra(ref_gray)
        curr_fea = self.homogan.fea_extra(curr_gray)
        
        # 2. 拼接
        forward_fea = torch.cat([ref_fea, curr_fea], dim=1)
        
        # 3. 预测权重
        weight_f = self.homogan.h_net(forward_fea)
        
        # 4. 生成光流 (带尺寸自适应)
        b, _, h, w = ref_gray.shape
        # 检查 basis 尺寸是否匹配当前输入
        if self.homogan.basis.shape[2] != h * w:
            # 如果不匹配，尝试调用 gen_basis 生成新的 (前提是 net.py 里有这个方法)
            # 如果 net.py 没有把 gen_basis 暴露出来，这里会报错，需要手动去 net.py 把它变成类方法
            try:
                basis = self.homogan.gen_basis(h, w).to(forward_fea.device)
            except AttributeError:
                # 备选方案：如果无法重新生成，就只能报错提示
                raise RuntimeError(f"输入尺寸 {h}x{w} 与 HomoGAN 预设 basis 尺寸不匹配，且未找到 gen_basis 方法。")
        else:
            basis = self.homogan.basis.to(forward_fea.device)
            
        H_flow = (basis * weight_f).sum(1).reshape(b, 2, h, w)
        return H_flow

    def forward(self, current_frame, ref_frame):
        curr_gray = self.rgb_to_gray(current_frame)
        ref_gray = self.rgb_to_gray(ref_frame)

        flow = self.get_homogan_flow(ref_gray, curr_gray)
        predicted_frame = get_warp_flow(ref_frame, flow)
        
        residual = current_frame - predicted_frame
        residual_shifted = (residual / 2.0) + 0.5 
        
        mamba_out = self.mamba_codec(residual_shifted)
        
        rec_residual = (mamba_out['x_hat'] - 0.5) * 2.0
        rec_frame = predicted_frame + rec_residual
        
        return {
            "rec_frame": rec_frame,
            "likelihoods": mamba_out.get('likelihoods', None),
            "predicted_frame": predicted_frame,
            "flow": flow
        }