import torch
import torch.nn as nn
import torch.nn.functional as F  # <--- 补上了这行关键引用
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

# 引用现有项目中的模块
# 请确保 VSS_module.py 和 MambaIC.py 在同一目录下
from mamba_models.VSS_module import conv, deconv
from mamba_models.MambaIC import VSSBlock, CheckboardMaskedConv2d, SWAtten, MambaIC

class MambaIC_NoCAM(MambaIC):
    """
    MambaIC No-CAM 版本
    - 移除了: Channel-wise Autoregressive (CAM) -> 通道并行处理
    - 保留了: VSS Backbone (Mamba), Spatial Context (Checkerboard), WLA (Window Attention)
    - 速度: 比原版快 (2-pass vs 5-pass)，比 Lite 版慢 (2-pass vs 1-pass)
    - 精度: 比 Lite 版高
    """
    def __init__(self, depths=[2, 2, 9, 2], drop_path_rate=0.1, N=128, M=320, **kwargs):
        # 调用父类初始化
        # 注意：这里传入 num_slices=5 是为了骗过父类的 assert，后面我们会重写
        super().__init__(depths, drop_path_rate, N, M, num_slices=5, **kwargs)
        
        # --- 核心修改：强制 num_slices = 1 ---
        self.num_slices = 1 
        num_per_slice = M # 所有通道一次性处理
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # [保留] Backbone 部分 (g_a, h_a, h_s, g_s) 不需要动，直接复用父类的
        
        # --- [重写] 上下文模型组件 (适配 1 个 Slice) ---
        
        # 1. 空间上下文预测 (Checkerboard)
        # 输入 M 通道，输出 2*M (Mean, Scale params)
        self.context_prediction = nn.ModuleList([
            CheckboardMaskedConv2d(
                M, 2 * M, kernel_size=5, padding=2, stride=1
            )
        ])

        # 2. 上下文融合模块 (Context VSS)
        # 原版有5个，这里只需要1个。
        c_depths = [2] 
        context_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(c_depths))]
        
        # 定义唯一的 Context VSS Block
        # hidden_dim 设置为足够容纳输入特征: Hyper特征(2*N) + 空间预测特征(2*M)
        in_dim = 2*N + 2*M 
        
        self.context_vss = nn.ModuleList([
            nn.Sequential(
                *[VSSBlock(hidden_dim=in_dim, drop_path=context_dpr[i], 
                           norm_layer=nn.LayerNorm, ssm_d_state=16, ssm_ratio=2.0)
                  for i in range(c_depths[0])],
                conv(in_dim, 2*M, kernel_size=3, stride=1) # 输出最终参数特征
            )
        ])

        # 3. WLA (Window-based Local Attention) 和 参数变换
        # 同样只需要 1 组
        self.atten_mean = nn.ModuleList([
            nn.Sequential(SWAtten(M, M, 16, self.window_size, 0, inter_dim=128))
        ])
        self.atten_scale = nn.ModuleList([
            nn.Sequential(SWAtten(M, M, 16, self.window_size, 0, inter_dim=128))
        ])

        # 参数投影层 (Transforms)
        self.cc_mean_transforms = nn.ModuleList([
            nn.Sequential(
                conv(2*M, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, M, stride=1, kernel_size=3),
            )
        ])
        self.cc_scale_transforms = nn.ModuleList([
            nn.Sequential(
                conv(2*M, 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, M, stride=1, kernel_size=3),
            )
        ])
        
        # 重新初始化熵模型以防万一
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        """
        No-CAM 的前向传播 (训练模式)
        """
        # 1. Main Encode
        y = self.g_a(x)
        
        # 2. Hyper Encode
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        # 3. Get Hyper Parameters (从 z 恢复的基础特征)
        hyper_feats_mean = self.h_mean_s(z_hat) # [B, 2*N, H, W]
        hyper_feats_scale = self.h_scale_s(z_hat)
        
        # 4. Checkerboard Context Prediction (空间上下文)
        
        # --- 计算参数 (Parameters Estimation) ---
        
        # 4.1 空间上下文特征
        y_masked = y # CheckboardMaskedConv2d 会在内部处理 mask
        ctx_params = self.context_prediction[0](y_masked) # [B, 2*M, H, W]
        
        # 4.2 融合 Hyper 和 Spatial
        # 为了安全起见，检查尺寸是否匹配 (F.interpolate 需要 F 定义)
        if hyper_feats_mean.shape[2:] != ctx_params.shape[2:]:
             hyper_feats_mean = F.interpolate(hyper_feats_mean, size=ctx_params.shape[2:], mode='nearest')
             hyper_feats_scale = F.interpolate(hyper_feats_scale, size=ctx_params.shape[2:], mode='nearest')

        # 拼接输入给 Context VSS
        fusion_input = torch.cat([hyper_feats_mean, ctx_params], dim=1)
        
        # 通过 VSS Block 提取上下文特征
        ctx_feats = self.context_vss[0](fusion_input) # [B, 2*M, H, W]
        
        # 4.3 生成最终 Mean 和 Scale
        mu = self.cc_mean_transforms[0](ctx_feats)
        sigma = self.cc_scale_transforms[0](ctx_feats)
        
        # 通过 WLA (Window Attention) Refinement
        mu = self.atten_mean[0](mu)
        sigma = self.atten_scale[0](sigma)
        
        # 5. Gaussian Conditional (熵编码/解码)
        y_hat, y_likelihoods = self.gaussian_conditional(y, sigma, means=mu)
        
        # 6. Main Decode
        x_hat = self.g_s(y_hat)
        
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods,
            },
        }

    # 用于推理的 compress 函数 (2-Pass 解码)
    # 注意：如果你只是训练，不需要实现这里。如果要在无人机上跑推理，需要补充 Checkerboard 解码逻辑。
    def compress(self, x):
        pass