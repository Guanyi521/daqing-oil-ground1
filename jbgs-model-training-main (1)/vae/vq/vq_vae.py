# (P, Sw, Ψ) 动态场的向量量化变分自编码器 QA-VAE

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from vae.layers import (
    Conv3dCylindrical,
    Interpolate3DCylindrical, 
    AttentionBlock3D, 
    VQEncoderPhysicsScaler,
    VQDecoderPhysicsScaler, 
    AutoPhysicsScaler,
    ResNetBlock3D
)


class SphericalVQ(nn.Module):
    """
    支持 EMA (解决 Codebook 坍塌) 且输出 Unreduced Loss (支持空间加权) 的 VQ 层
    这对于后续接入 Latent Diffusion 至关重要！
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        # 🌟 注意：这里 self.actual_vocab_size (num_embeddings 个码 + 1个Padding)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.actual_vocab_size = num_embeddings + 1
        self.commitment_cost = commitment_cost

        # padding_idx=0: 索引 0 自动映射为全 0 向量，且梯度不更新
        self.embedding = nn.Embedding(self.actual_vocab_size, self.embedding_dim, padding_idx=0)
        
        # 1. 暴力初始化整个词表 (范围基于真实的物理码个数 512 会更科学)
        init_bound = 1.0 / num_embeddings 
        self.embedding.weight.data.uniform_(-init_bound, init_bound)
        
        # 2. 强行把 Padding 位置归零，确保绝对纯净
        with torch.no_grad():
            self.embedding.weight[0].fill_(0.0)

    def forward(self, inputs, mask_latent=None):
        B, C, nz, n_theta, n_R = inputs.shape
        # (B, C, nz, n_theta, n_R) -> (B, nz, n_theta, n_R, C)
        inputs_permuted = inputs.permute(0, 2, 3, 4, 1).contiguous()
        flat_inputs = inputs_permuted.view(-1, self.embedding_dim)

        # ===================================================================
        # 🌟 核心升级：Spherical VQ (L2 归一化)
        # ===================================================================
        # 1. 对网络输出进行归一化 (增加 eps 防止梯度除零异常)
        flat_inputs_norm = F.normalize(flat_inputs, p=2, dim=1, eps=1e-12)
        
        # 2. 对密码本进行归一化 (第 0 行本来是全0，加 eps 后无限接近 0，不影响后续运算)
        weight_norm = F.normalize(self.embedding.weight, p=2, dim=1, eps=1e-12)

        # 3. 在超球面上计算距离 (此时欧氏距离完全等价于负的余弦相似度)
        distances = (torch.sum(flat_inputs_norm**2, dim=1, keepdim=True) 
                    + torch.sum(weight_norm**2, dim=1)
                    - 2 * torch.matmul(flat_inputs_norm, weight_norm.t()))
        
        # 4. 获取 VQ 码：在 1~1024 的码本中找最近的 (跳过第 0 列 Padding)
        encoding_indices = torch.argmin(distances[:, 1:], dim=1).unsqueeze(1) + 1

        # 5. 处理无效区域 (石头)
        if mask_latent is not None:
            mask_flat = mask_latent.view(-1, 1).bool()
            encoding_indices.masked_fill_(~mask_flat, 0)

        # 6. 查表获取量化特征，注意：这里必须使用归一化后的密码本！
        # F.embedding 是 nn.Embedding 的函数式调用，允许我们传入自定义的 weight
        quantized_norm = F.embedding(encoding_indices, weight_norm).view(inputs_permuted.shape)

        # 将 Normalized 的输入还原为 5D 形状，用于计算 Loss
        flat_inputs_norm_5d = flat_inputs_norm.view(inputs_permuted.shape)

        # ===================================================================
        # 7. 计算 Loss (在超球面上计算，彻底解决高维自由度过大的问题)
        # ===================================================================
        vq_loss_unreduced = (quantized_norm.detach() - flat_inputs_norm_5d)**2 + \
                            self.commitment_cost * (quantized_norm - flat_inputs_norm_5d.detach())**2
        vq_loss_spatial = vq_loss_unreduced.mean(dim=-1)

        if mask_latent is not None:
            mask_bool = mask_latent.squeeze(1).bool()
            vq_loss_spatial = vq_loss_spatial * mask_bool # 石头不产生 Loss

        # ===================================================================
        # 8. Straight-Through Estimator (绕过不可导的量化过程)
        # ===================================================================
        # 注意：这里我们让超球面上的 Normalized 输入穿透过去！
        # 因为 Decoder 末端有 PhysicsScaler(unscaler)，所以 Decoder 完全可以处理均值为0、方差为1的球面特征！
        quantized_out = flat_inputs_norm_5d + (quantized_norm - flat_inputs_norm_5d).detach()
        quantized_out = quantized_out.permute(0, 4, 1, 2, 3).contiguous()

        # 9. 物理边界阻断：确保送给 Decoder 的特征绝对纯净
        if mask_latent is not None:
            quantized_out = quantized_out * mask_latent
        
        # 还原 indices 形状供 Transformer 离线保存
        spatial_indices = encoding_indices.view(B, nz, n_theta, n_R)

        return quantized_out, vq_loss_spatial, spatial_indices

class VQEncoder3D(nn.Module):
    """
    完美适配 Latent Diffusion 的 Encoder
    集成环形卷积 + 注意力瓶颈 + Swish 激活
    """
    def __init__(self, 
                 in_channels=5, 
                 hidden_dim=64, latent_dim=64, 
                 num_res_blocks=2,
                 num_attn_heads=8,
                 compress_shape=(2, 2, 2),
    ):
        super().__init__()

       # 注入物理缩放器
        # self.scaler = VQEncoderPhysicsScaler(in_channels)
        self.scaler = AutoPhysicsScaler(in_channels)
        
        self.in_conv = Conv3dCylindrical(in_channels, hidden_dim, kernel_size=3, stride=1)

        # 下采样前加持 ResNet 容量
        self.res_blocks1 = nn.ModuleList([ResNetBlock3D(hidden_dim) for _ in range(num_res_blocks)])
        
        # 对于给定的压缩比 s:
        # - 如果 s = 2，标准重叠下采样最优解是: kernel=4, padding=1
        # - 如果 s = 1 (该维度不压缩): kernel=3, padding=1
        # - 如果是其他压缩比 (如 s=3): 粗暴点直接用 kernel=s, padding=0
        k_size = tuple(4 if s == 2 else (3 if s == 1 else s) for s in compress_shape)
        p_size = tuple(1 if s == 2 else (1 if s == 1 else 0) for s in compress_shape)

        self.down_block = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(), # SiLU (Swish) 在生成模型中比 ReLU 更平滑，梯度更好
            Conv3dCylindrical(
                hidden_dim, hidden_dim * 2, 
                kernel_size=k_size, 
                stride=compress_shape,
                padding=p_size
            )
        )

        # 瓶颈处加持 ResNet 容量
        self.res_blocks2 = nn.ModuleList([ResNetBlock3D(hidden_dim * 2) for _ in range(num_res_blocks)])
        

        self.attention = AttentionBlock3D(hidden_dim * 2, num_attn_heads)

        self.bottleneck_out = nn.Sequential(
            nn.GroupNorm(8, hidden_dim * 2),
            nn.SiLU(),
            # 修改处：输出通道数为 latent_dim * 2 (mu 和 logvar)
            Conv3dCylindrical(hidden_dim * 2, latent_dim * 2, kernel_size=3, stride=1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 进网络前，先把 (P, Sw, Ψ) 缩放到大致相似的量级 N(0, 1) 附近
        x_scaled = self.scaler(x)
        
        # 用 mask 切掉石头里的无效数值带来的干扰
        h = self.in_conv(x_scaled * mask) * mask

        # 残差层加强
        for res_block in self.res_blocks1:
            h = res_block(h, mask)

        # 下采样
        h = self.down_block(h)

        # mask 需要跟着下采样
        latent_mask = F.interpolate(mask.float(), size=h.shape[2:], mode='nearest')

        # 第二道阻断
        h = h * latent_mask 

        # 残差加强
        for res_block in self.res_blocks2:
            h = res_block(h, latent_mask)

        # 注意力层
        h = self.attention(h, latent_mask)
        z = self.bottleneck_out(h)

        # 沿通道维度劈开，一半是均值，一半是对数方差
        mu, logvar = torch.chunk(z, 2, dim=1)
        return mu*latent_mask, logvar*latent_mask, latent_mask

class VQDecoder3D(nn.Module):
    """
    完美适配 Latent Diffusion 的 Decoder
    使用 Resize-Conv 替代反卷积，集成环形卷积 + 注意力瓶颈
    """
    def __init__(self, 
                 latent_dim=64, hidden_dim=64, 
                 out_channels=5, 
                 num_res_blocks=2, 
                 num_attn_heads=8,
                 compress_shape=(2, 2, 2)
    ):
        super().__init__()

        self.in_conv = Conv3dCylindrical(latent_dim, hidden_dim * 2, kernel_size=3, stride=1)
        
        # 上采样加持残差网络 1
        self.res_blocks1 = nn.ModuleList([ResNetBlock3D(hidden_dim * 2) for _ in range(num_res_blocks)])

        self.attention = AttentionBlock3D(hidden_dim * 2, num_attn_heads)

        # 🌟 实例化环形插值模块
        self.circular_interpolate = Interpolate3DCylindrical(compress_shape)

        # 上采样块 (Resize-Conv 避免棋盘效应)
        self.up_conv = Conv3dCylindrical(hidden_dim * 2, hidden_dim, kernel_size=3, stride=1)

        # 上采样加持残差网络 2
        self.res_blocks2 = nn.ModuleList([ResNetBlock3D(hidden_dim) for _ in range(num_res_blocks)])
        
        self.out_block = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            Conv3dCylindrical(hidden_dim, out_channels, kernel_size=3, stride=1)
        )
        # 挂载 Decoder 专属的带 Softmax 约束缩放器
        # self.unscaler = VQDecoderPhysicsScaler(out_channels, sat_indices=[1, 2, 3])
        self.unscaler = AutoPhysicsScaler(out_channels)

    def forward(self, z_q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        latent_mask = F.interpolate(mask.float(), size=z_q.shape[2:], mode='nearest')

        h = self.in_conv(z_q) * latent_mask

        # 上采样加持残差网络 1
        for res_block in self.res_blocks1:
            h = res_block(h, latent_mask)

        h = self.attention(h, latent_mask)

        # 🌟 使用完美的环形物理插值替代原生插值
        h_up = self.circular_interpolate(h, target_size=mask.shape[2:])

        # 再经过自带环形 Padding 的 Conv3dCircularTheta
        h_up = self.up_conv(h_up) * mask

        # 上采样加持残差网络 2
        for res_block in self.res_blocks2:
            h_up = res_block(h_up, mask)
        
        # 注意：此处不加激活函数，让网络自由输出物理场的负数或大数值
        out = self.out_block(h_up)

        out = self.unscaler(out)

        return out * mask
    
class VQVAE3D(pl.LightningModule):
    """
    封装好的 Lightning VQ-VAE 模块。
    支持第一阶段的完全独立预训练。
    """
    def __init__(self, 
                 in_channels=5, 
                 hidden_dim=32, latent_dim=64, 
                 num_embeddings=512, 
                 commitment_cost=0.25, 
                 num_res_blocks=2,
                 num_attn_heads=8,
                 compress_shape=(2, 2, 2)
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = VQEncoder3D(in_channels, hidden_dim, latent_dim, num_res_blocks, num_attn_heads, compress_shape)
        self.vq_layer = SphericalVQ(num_embeddings, latent_dim, commitment_cost)
        self.decoder = VQDecoder3D(latent_dim, hidden_dim, in_channels, num_res_blocks, num_attn_heads, compress_shape)

    
    def _compute_hd_loss(self, recon, x, mask, locs_3d):
        # TODO: 根据后续数据集的具体格式补齐逻辑
        return torch.tensor(0.0, device=recon.device)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧：z = mu + eΨlon * sigma"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        """更新 forward，返回 mu 和 logvar 给 Diffusion 使用"""
        mu, logvar, latent_mask = self.encoder(x, mask)
        # 采样得到连续的潜变量 z_e
        z_e = self.reparameterize(mu, logvar)
        z_q, vq_loss_spatial, spatial_indices = self.vq_layer(z_e, latent_mask)
        x_recon = self.decoder(z_q, mask)
        return x_recon, vq_loss_spatial, spatial_indices, mu, logvar

    @torch.no_grad()
    def encode_to_indices(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        """提供接口给 Diffusion：返回 mu, logvar 和 VQ Token"""
        mu, logvar, latent_mask = self.encoder(x, mask)
        z_e = mu # 推理时直接使用均值，不加噪声
        _, _, indices = self.vq_layer(z_e, latent_mask)
        return indices, mu, logvar

    @torch.no_grad()
    def decode_from_indices(self, indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        供 Stage 2 (Transformer) 调用的接口：根据预测出的 Token ID 还原物理场。
        Args:
            indices: 整数张量 (B, nz/2, n_theta/2, n_R/2)
        Returns:
            x_recon: 重建的物理场 (B, 5, nz, n_theta, n_R)
        """
        # 1. 必须先获取超球面上的 Normalized 密码本
        weight_norm = F.normalize(self.vq_layer.embedding.weight, p=2, dim=1, eps=1e-12)
        
        # 2. 用函数式接口查表，获取模长为 1 的特征
        quantized = F.embedding(indices, weight_norm) 
        
        # 3. 维度重排并送入解码器
        quantized = quantized.permute(0, 4, 1, 2, 3).contiguous()
        return self.decoder(quantized, mask)

    def training_step(self, batch, batch_idx):
        # 你的 batch 必须多返回一个 hard_data_locs
        x, hard_data_locs, mask = batch 

        mu, logvar, latent_mask = self.encoder(x, mask)
        z_e = self.reparameterize(mu, logvar)
        z_q, vq_loss_spatial, _ = self.vq_layer(z_e, latent_mask)
        x_recon = self.decoder(z_q, mask)
    
        # ==========================================
        #  物理场重建损失 
        # ==========================================
        recons_loss_spatial = F.l1_loss(x_recon, x, reduction='none').mean(dim=1) 
        
        # ==========================================
        #  生成径向衰减权重 W(r)
        # ==========================================
        n_R_original = x.shape[-1]
        n_R_latent = mu.shape[-1]
        
        r_indices_orig = torch.arange(n_R_original, device=x.device, dtype=torch.float32)
        W_R_orig = torch.exp(-0.2 * r_indices_orig).view(1, 1, 1, n_R_original)
        
        r_indices_latent = torch.arange(n_R_latent, device=x.device, dtype=torch.float32)
        W_R_latent = torch.exp(-0.2 * r_indices_latent).view(1, 1, 1, n_R_latent)
        
        # ==========================================
        #  包含 W(R) 径向模糊的 KL Loss 与 VQ Loss
        # ==========================================
        # KL Loss 公式：0.5 * (mu^2 + exp(logvar) - logvar - 1)
        # 对通道维度求和或求均值，保留空间维度 (B, nz, n_theta, n_R)
        kl_loss_spatial = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1, dim=1)
        
        # W_R 最好也规范为 5D 形状，彻底杜绝广播歧义
        W_R_orig_5d = W_R_orig.view(1, 1, 1, 1, -1)
        W_R_latent_5d = W_R_latent.view(1, 1, 1, 1, -1)

        # 补齐 unsqueeze(1)
        valid_weighted_kl_loss = (kl_loss_spatial.unsqueeze(1) * W_R_latent_5d * latent_mask).sum() / (latent_mask.sum() + 1e-8)
        
        # valid_weighted_vq_loss 你之前写了 unsqueeze(1)，那是对的，这里规范化 W_R
        valid_weighted_vq_loss = (vq_loss_spatial.unsqueeze(1) * W_R_latent_5d * latent_mask).sum() / (latent_mask.sum() + 1e-8)
        
        # 补齐 unsqueeze(1)
        valid_weighted_recons_loss = (recons_loss_spatial.unsqueeze(1) * W_R_orig_5d * mask).sum() / (mask.sum() + 1e-8)

        # ==========================================
        #  硬数据约束 (处理稀疏的井底/井口数据)
        # hard_data_locs.shape = (B, C, n_locs, 3) 代表 (z_k, theta_j, R_i)
        # ==========================================
        # 调用我们封装好的高阶物理约束函数！
        # 注意: hard_data_locs 如果多了一个冗余通道，这里取 [:, 0, :, :] 变为 (B, n_locs, 3)
        locs_3d = hard_data_locs[:, 0, :, :] if hard_data_locs.dim() == 4 else hard_data_locs
        hd_loss, hd_loss_dict = self._compute_hd_loss(x_recon, x, mask, locs_3d)
        hd_weight = 10.0
        
        # ==========================================
        # 总损失
        # ==========================================
        kl_weight = 1e-4 # 通常 KL Loss 的权重需要设置得很小，防止过早后验崩塌
        loss_tot = valid_weighted_recons_loss + valid_weighted_vq_loss + (kl_weight * valid_weighted_kl_loss) + (hd_weight * hd_loss)
        

        self.log('train/recons_loss', valid_weighted_recons_loss)
        self.log('train/vq_loss', valid_weighted_vq_loss)
        self.log('train/kl_loss', valid_weighted_kl_loss)

        # 日志记录 (直接解包字典)
        for key, val in hd_loss_dict.items():
            self.log(f'train/{key}', val)
        self.log('train_loss', loss_tot)
        
        return loss_tot

    def configure_optimizers(self):
        # 经验之谈：VQ-VAE 对学习率比较敏感，通常 1e-3 到 2e-4 比较合适
        return torch.optim.Adam(self.parameters(), lr=2e-4)


if __name__ == "__main__":
    print("="*60)
    print("🚀 初始化 VQVAE3D 模型并执行前向校验...")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化你的参数，in_channels=5 对应 [P, Cw, Co, Cg, Ψ]
    model = VQVAE3D(
        in_channels=5, 
        hidden_dim=32, 
        latent_dim=64, 
        num_embeddings=1024, 
        num_res_blocks=2, 
        num_attn_heads=8,
        compress_shape=(2, 2, 2)
    ).to(device)
    
    # 模拟输入 Tensor (B, C, nz, n_theta, n_R)
    B, C, nz, n_theta, n_R = 2, 5, 20, 20, 10
    x = torch.randn(B, C, nz, n_theta, n_R).to(device)
    mask = torch.ones(B, 1, nz, n_theta, n_R).to(device)
    
    print(f"📥 1. 输入数据形状 (x): {x.shape}")
    print(f"   [通道定义]: C=0(P), C=1(Cw), C=2(Co), C=3(Cg), C=4(Ψ)")
    
    # 拆解单步执行，方便观测特征变化
    mu, logvar, latent_mask = model.encoder(x, mask)
    print(f"\n🧠 2. 编码器输出:")
    print(f"   mu 形状: {mu.shape}")
    print(f"   logvar 形状: {logvar.shape}")
    
    z_e = model.reparameterize(mu, logvar)
    z_q, vq_loss_spatial, spatial_indices = model.vq_layer(z_e, latent_mask)
    print(f"\n🔮 3. 量化层输出 (Spherical VQ):")
    print(f"   量化特征 z_q 形状: {z_q.shape}")
    print(f"   离散 Codebook ID 形状: {spatial_indices.shape}")
    print(f"   VQ 空间损失形状: {vq_loss_spatial.shape}")
    
    x_recon = model.decoder(z_q, mask)
    print(f"\n📤 4. 解码器与缩放器输出:")
    print(f"   重建场 x_recon 形状: {x_recon.shape}")
    
    # 校验通道 Softmax 硬约束是否生效
    S_sum = x_recon[:, 1:4, ...].sum(dim=1)
    print(f"\n✅ 5. 校验物理硬约束 (Cw + Co + Cg = 1):")
    print(f"   三个组分饱和度相加后的最小值: {S_sum.min().item():.6f}")
    print(f"   三个组分饱和度相加后的最大值: {S_sum.max().item():.6f}")
    
    if torch.allclose(S_sum, torch.ones_like(S_sum), atol=1e-5):
        print("   -> 物理硬约束验证通过！网络已不可能输出非物理越界组分。")
    else:
        print("   -> 警告：物理硬约束失效！")
    print("="*60)