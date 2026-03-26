import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 确保引入上一轮我们写好的工业级算子
from vae.layers import (
    Conv3dCylindrical, 
    Interpolate3DCylindrical, 
    AttentionBlock3D, 
    ResNetBlock3D,
    MinMaxScaler,
    LogScaler
)

# ==========================================
# 1. 基础 VAE 编码器与解码器 (注入 Mask 与 Scaler)
# ==========================================

class GeoEncoder3D(nn.Module):
    def __init__(self, scaler: nn.Module, in_channels=1, hidden_dim=64, latent_channels=4, compress_shape=(2, 2, 2), num_res_blocks=1):
        super().__init__()
        self.scaler = scaler
        self.in_conv = Conv3dCylindrical(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.res1 = nn.ModuleList([ResNetBlock3D(hidden_dim) for _ in range(num_res_blocks)])
        
        k_size = tuple(4 if s == 2 else 3 for s in compress_shape)
        p_size = tuple(1 if s == 2 else 1 for s in compress_shape)
        
        self.down = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            Conv3dCylindrical(hidden_dim, hidden_dim * 2, kernel_size=k_size, stride=compress_shape, padding=p_size)
        )
        
        self.res2 = nn.ModuleList([ResNetBlock3D(hidden_dim * 2) for _ in range(num_res_blocks)])
        self.attention = AttentionBlock3D(hidden_dim * 2)
        self.out_conv = Conv3dCylindrical(hidden_dim * 2, latent_channels * 2, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        h = self.scaler.scale(x)
        h = self.in_conv(h * mask) * mask
        for res in self.res1:
            h = res(h, mask)
        h = self.down(h)
        
        # 下采样 Mask
        latent_mask = F.interpolate(mask.float(), size=h.shape[2:], mode='nearest')
        
        h = h * latent_mask
        for res in self.res2:
            h = res(h, latent_mask)
        h = self.attention(h, latent_mask)
        
        z = self.out_conv(h) * latent_mask
        mu, logvar = torch.chunk(z, 2, dim=1)
        return mu * latent_mask, logvar * latent_mask, latent_mask


class GeoDecoder3D(nn.Module):
    def __init__(self, scaler: nn.Module, latent_channels=4, hidden_dim=64, out_channels=1, compress_shape=(2, 2, 2), num_res_blocks=1):
        super().__init__()
        self.scaler = scaler
        self.in_conv = Conv3dCylindrical(latent_channels, hidden_dim * 2, kernel_size=3, padding=1)
        self.attention = AttentionBlock3D(hidden_dim * 2)
        self.res1 = nn.ModuleList([ResNetBlock3D(hidden_dim * 2) for _ in range(num_res_blocks)])
        
        self.circular_interpolate = Interpolate3DCylindrical(compress_shape)
        self.up_conv = Conv3dCylindrical(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1)
        
        self.res2 = nn.ModuleList([ResNetBlock3D(hidden_dim) for _ in range(num_res_blocks)])
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            Conv3dCylindrical(hidden_dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, z: torch.Tensor, target_shape: tuple, mask: torch.Tensor) -> torch.Tensor:
        latent_mask = F.interpolate(mask.float(), size=z.shape[2:], mode='nearest')
        
        h = self.in_conv(z) * latent_mask
        h = self.attention(h, latent_mask)
        for res in self.res1:
            h = res(h, latent_mask)

        
        h_up = self.circular_interpolate(h, target_shape)
        h_up = self.up_conv(h_up) * mask
        
        for res in self.res2:
            h_up = res(h_up, mask)
        out_scaled = self.out_conv(h_up)
        
        return self.scaler.unscale(out_scaled) * mask


# ==========================================
# 2. 基础 Lightning VAE 模块 (基类)
# ==========================================

class GeoVAE3D(pl.LightningModule):
    # 🌟 修复: 加入了 scaler, hd_weight，并对齐初始化
    def __init__(self, scaler: nn.Module, in_channels=1, hidden_dim=64, latent_channels=4, compress_shape=(2, 2, 2), num_res_blocks=1, kl_weight=1e-5, hd_weight=10.0, decay_rate=0.2):
        super().__init__()
        self.save_hyperparameters(ignore=['scaler'])
        self.scaler = scaler
        self.encoder = GeoEncoder3D(scaler, in_channels, hidden_dim, latent_channels, compress_shape, num_res_blocks)
        self.decoder = GeoDecoder3D(scaler, latent_channels, hidden_dim, in_channels, compress_shape, num_res_blocks)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        target_shape = x.shape[2:]
        mu, logvar, latent_mask = self.encoder(x, mask)
        z = self.reparameterize(mu, logvar)   #这里是潜特征，用于和pipeline中的静态特征联合   B,C, lat_nz, lat_nθ, lat_R
        x_recon = self.decoder(z, target_shape, mask)
        return x_recon, mu, logvar, latent_mask
    
    # 替换原来的 _compute_hd_loss 函数
    def _compute_hd_loss(self, recon, target, mask, hd_locs):
        """
        利用 PyTorch 高级索引直接沿 (theta, R) 坐标垂直穿透提取 Z 轴。
        不产生任何广播带来的显存开销！
        """
        B, C, nz, nθ, nR = recon.shape
        total_hd_loss = 0.0
        total_valid_pts = 0.0

        for b in range(B):
            # 找到该样本下有效的井点 (第一位标志为 1.0)
            valid_mask = hd_locs[b, :, 0] == 1.0
            valid_pts = hd_locs[b, valid_mask] # 形状: (N, 3)
            
            if valid_pts.shape[0] == 0:
                continue
                
            idx_t = valid_pts[:, 1].long()
            idx_r = valid_pts[:, 2].long()
            
            # 极速穿透提取: 结果形状为 (nz, N)
            r_col = recon[b, 0, :, idx_t, idx_r]
            t_col = target[b, 0, :, idx_t, idx_r]
            m_col = mask[b, 0, :, idx_t, idx_r]
            
            # 计算这一批井柱的 MSE
            mse = F.mse_loss(r_col, t_col, reduction='none')
            total_hd_loss += (mse * m_col).sum()
            total_valid_pts += m_col.sum()
            
        if total_valid_pts > 0:
            return total_hd_loss / (total_valid_pts + 1e-8)
        else:
            return torch.tensor(0.0, device=recon.device)
        
        
    def training_step(self, batch, batch_idx):
        geo_field, hard_data_points, mask = batch
        recon, z_mu, z_sigma, latent_mask = self(geo_field, mask)
        
        # ==========================================
        # 🌟 物理权重 W(r) 对齐：越往外圈，体积越大，权重衰减
        # ==========================================
        n_R_original = geo_field.shape[-1]
        n_R_latent = z_mu.shape[-1]
        
        r_indices_orig = torch.arange(n_R_original, device=self.device, dtype=torch.float32)
        W_R_orig = torch.exp(-self.hparams.decay_rate * r_indices_orig).view(1, 1, 1, 1, n_R_original)
        
        r_indices_latent = torch.arange(n_R_latent, device=self.device, dtype=torch.float32)
        W_R_latent = torch.exp(-self.hparams.decay_rate * r_indices_latent).view(1, 1, 1, 1, n_R_latent)
        
        # 1. 带物理权重的 L2 重建损失
        recon_loss = F.mse_loss(recon, geo_field, reduction='none')
        valid_weighted_recon_loss = (recon_loss * W_R_orig * mask).sum() / (mask.sum() + 1e-8)
        
        # 2. 带物理权重的 KL 散度损失
        kl = 0.5 * (z_mu.pow(2) + z_sigma.exp() - z_sigma - 1)
        valid_weighted_kl_loss = (kl * W_R_latent * latent_mask).sum() / (latent_mask.sum() + 1e-8)
        
        # 3. 硬数据损失 (HD Loss)
        hd_loss = self._compute_hd_loss(recon, geo_field, mask, hard_data_points)
        
        loss = valid_weighted_recon_loss + self.hparams.kl_weight * valid_weighted_kl_loss + self.hparams.hd_weight * hd_loss
        
        self.log('train/recon_loss', valid_weighted_recon_loss)
        self.log('train/kl_loss', valid_weighted_kl_loss)
        self.log('train/hd_loss', hd_loss)
        self.log('train_loss', loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        geo_field, hard_data_points, mask = batch
        recon, z_mu, z_sigma, latent_mask = self(geo_field, mask)
        
        n_R_original = geo_field.shape[-1]
        n_R_latent = z_mu.shape[-1]
        
        r_indices_orig = torch.arange(n_R_original, device=self.device, dtype=torch.float32)
        W_R_orig = torch.exp(-self.hparams.decay_rate * r_indices_orig).view(1, 1, 1, 1, n_R_original)
        
        r_indices_latent = torch.arange(n_R_latent, device=self.device, dtype=torch.float32)
        W_R_latent = torch.exp(-self.hparams.decay_rate * r_indices_latent).view(1, 1, 1, 1, n_R_latent)
        
        recon_loss = F.mse_loss(recon, geo_field, reduction='none')
        valid_weighted_recon_loss = (recon_loss * W_R_orig * mask).sum() / (mask.sum() + 1e-8)
        
        kl = 0.5 * (z_mu.pow(2) + z_sigma.exp() - z_sigma - 1)
        valid_weighted_kl_loss = (kl * W_R_latent * latent_mask).sum() / (latent_mask.sum() + 1e-8)
        
        hd_loss = self._compute_hd_loss(recon, geo_field, mask, hard_data_points)
        
        loss = valid_weighted_recon_loss + self.hparams.kl_weight * valid_weighted_kl_loss + self.hparams.hd_weight * hd_loss
        
        # sync_dist=True 保证在多 GPU 验证时数据能正确同步
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/recon_loss', valid_weighted_recon_loss, sync_dist=True)
        self.log('val/kl_loss', valid_weighted_kl_loss, sync_dist=True)
        self.log('val/hd_loss', hd_loss, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        # 🌟 修复：补充优化器
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    @torch.no_grad()
    def encode_latent(self, x: torch.Tensor, mask: torch.Tensor, use_mean: bool = True):
        mu, logvar, latent_mask = self.encoder(x, mask)
        z = mu if use_mean else self.reparameterize(mu, logvar)
        return z, latent_mask
    
# ==========================================
# 4. 特化物理场 VAE 模块 (继承机制完美解耦)
# ==========================================

class PoroVAE3D(GeoVAE3D):
    def __init__(self, 
                 poro_min, poro_max, 
                 hidden_dim=64, latent_channels=4, 
                 compress_shape=(2, 2, 2), 
                 num_res_blocks=1,
                 kl_weight=1e-5, hd_weight=10.0, 
                 decay_rate=0.2
    ):
        scaler = MinMaxScaler(min_val=poro_min, max_val=poro_max)
        super().__init__(
            scaler=scaler, in_channels=1, 
            hidden_dim=hidden_dim, latent_channels=latent_channels, 
            compress_shape=compress_shape, num_res_blocks=num_res_blocks,
            kl_weight=kl_weight, hd_weight=hd_weight,
            decay_rate=decay_rate
        )

class PermVAE3D(GeoVAE3D):
    def __init__(self, 
                 perm_min, perm_max, 
                 hidden_dim=64, latent_channels=4, 
                 compress_shape=(2, 2, 2), 
                 num_res_blocks=1,
                 kl_weight=1e-5, hd_weight=10.0, 
                 decay_rate=0.2
    ):
        scaler = LogScaler(min_val=perm_min, max_val=perm_max)
        super().__init__(
            scaler=scaler, in_channels=1, 
            hidden_dim=hidden_dim, latent_channels=latent_channels, 
            compress_shape=compress_shape, num_res_blocks=num_res_blocks,
            kl_weight=kl_weight, hd_weight=hd_weight, 
            decay_rate=decay_rate
        )


# ==========================================
# 🚀 极限网络测试模块
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("🚀 启动 GeoVAE3D (孔隙度与渗透率独立 VAE) 本地极限测试...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 当前运行设备: {device}")
    
    # --- 1. 模拟数据生成 ---
    B, C, nz, n_theta, n_R = 2, 1, 20, 20, 10
    
    # 模拟孔隙度数据 [0.05, 0.35]
    poro_field = torch.empty(B, C, nz, n_theta, n_R).uniform_(0.05, 0.35).to(device)
    # 模拟渗透率数据 [0.1, 5000] (对数正态分布特性)
    perm_field = torch.pow(10, torch.empty(B, C, nz, n_theta, n_R).uniform_(-1, 3.7)).to(device)
    
    # 构造边界 Mask
    mask = (torch.rand(B, 1, nz, n_theta, n_R) > 0.1).float().to(device)
    # 虚拟的 HD locs
    dummy_hd_points = torch.zeros(B, 10, 3).to(device)
    
    # --- 2. 实例化并测试孔隙度 VAE ---
    print("\n[测试 1]: PoroVAE3D (孔隙度)")
    poro_vae = PoroVAE3D(poro_min=0.0, poro_max=0.4, latent_channels=4).to(device)
    poro_vae.log = lambda *args, **kwargs: None # 屏蔽 logger 报错
    
    batch_poro = (poro_field, dummy_hd_points, mask)
    loss_poro = poro_vae.training_step(batch_poro, 0)
    print(f"✅ PoroVAE3D 前向与 Loss 计算通过! (Loss: {loss_poro.item():.4f})")
    
    # --- 3. 实例化并测试渗透率 VAE ---
    print("\n[测试 2]: PermVAE3D (渗透率)")
    perm_vae = PermVAE3D(perm_min=0.1, perm_max=10000.0, latent_channels=8).to(device)
    perm_vae.log = lambda *args, **kwargs: None
    
    batch_perm = (perm_field, dummy_hd_points, mask)
    loss_perm = perm_vae.training_step(batch_perm, 0)
    
    # 校验输出反算范围
    recon_perm, _, _, _ = perm_vae(perm_field, mask)
    
    print(f"✅ PermVAE3D 前向与 Loss 计算通过! (Loss: {loss_perm.item():.4f})")
    print(f"   [物理边界验证] 渗透率重建最小值: {recon_perm[mask.bool()].min().item():.2f} mD (理论应>=0.1)")
    print(f"   [物理边界验证] 渗透率重建最大值: {recon_perm[mask.bool()].max().item():.2f} mD")
    print("="*60)