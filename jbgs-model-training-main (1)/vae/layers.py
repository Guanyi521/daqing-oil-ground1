# layer struction for vqvae and vae_geo
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------
#   layer struction for vqvae and vae_geo 
# --------------------------------------------------------------------------------
class Conv3dCylindrical(nn.Module):
    """
    专为柱坐标系设计的 3D 卷积层 (支持多尺度/非对称压缩比)
    - Z 轴 (深度) 和 R 轴 (极径) 采用常规零填充 (Zero Padding)
    - Theta 轴 (方位角) 采用环形物理填充 (Circular Padding)
    """
    # 🌟 重点 1：将 padding 作为参数暴露出来，并默认设为 1
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, use_r_coord=False):
        super().__init__()
        self.use_r_coord = use_r_coord
        actual_in_channels = in_channels + 1 if use_r_coord else in_channels
        
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
        
        pad_Z, self.pad_theta, pad_R = self.padding
        
        # 🌟 优化 1：让 Conv 接管 Z 和 R 的补 0，省去 F.pad 的显存开销
        # nn.Conv3d 的 padding 顺序对应 (D, H, W) 即 (Z, Theta, R)
        self.conv = nn.Conv3d(
            actual_in_channels, out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=(pad_Z, 0, pad_R), # Theta 轴设为 0，交给我们手动 circular
            bias=bias
        )
        
        # 🌟 优化 2：注册持久化 buffer，避免每次前向传播重复计算
        self.register_buffer('r_coord', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, nz, n_theta, n_R = x.shape
        
        # 1. CoordConv
        if self.use_r_coord:
            # 只在第一次或输入 R 尺寸改变时生成 (避免多余计算)
            if self.r_coord is None or self.r_coord.shape[-1] != n_R:
                r_linear = torch.linspace(0, 1, n_R, device=x.device, dtype=x.dtype)
                # 使用平方根拉伸外圈特征是一个很好的先验，归一化到 [-1, 1]
                r_mapped = torch.sqrt(r_linear) * 2.0 - 1.0 
                # (1, 1, 1, 1, n_R)
                r_coord = r_mapped.view(1, 1, 1, 1, n_R)
                self.r_coord = r_coord
            
            # 扩展并拼接
            x = torch.cat([x, self.r_coord.expand(B, 1, nz, n_theta, n_R)], dim=1)

        # 2. 仅对 Theta 轴进行 Circular 补齐
        if self.pad_theta > 0:
            # F.pad 从右往左对应 (W, H, D) -> (R, Theta, Z)
            x = F.pad(x, pad=(0, 0, self.pad_theta, self.pad_theta, 0, 0), mode='circular')
            
        return self.conv(x)

class Interpolate3DCylindrical(nn.Module):
    """
    专为柱坐标系定制的物理插值模块。(上采样)
    融合了 Theta 轴的 Circular Padding 与 R 轴的等面积 (r^2) 映射插值。
    """

    def __init__(self, compress_shape: tuple):
        super().__init__()
        self.compress_shape = compress_shape
        # 🌟 注册 buffer 缓存 3D 网格，节省大量前向传播时间
        self.register_buffer('grid_3d', None)
        self.cached_in_shape = None
        self.cached_out_shape = None

    def forward(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        B, C, nz_lat, nθ_lat, nR_lat = x.shape
        out_nz, out_n_theta, out_n_R = target_size
        scale_Z, scale_theta, scale_R = self.compress_shape
        device = x.device

        # 检查是否可以直接使用缓存的网格
        if self.grid_3d is None or self.cached_in_shape != x.shape[2:] or self.cached_out_shape != target_size:
            
            # ==========================================================
            # 1. 修正 Theta 网格: 直接映射到 Padding 后的正确区间，告别输出裁切！
            # ==========================================================
            # 使用 align_corners=True 更易于精准控制坐标系
            grid_Z = torch.linspace(-1, 1, out_nz, device=device)
            
            if scale_theta > 1:
                # 如果输入左右各 pad 了 1 层，输入总宽变为 nθ_lat + 2
                # 原数据在 idx [1, nθ_lat] 之间。我们要输出的 out_n_theta 点必须落在这个区间！
                t_out_val = torch.linspace(0, nθ_lat - 1, steps=out_n_theta, device=device)
                t_idx = t_out_val + 1 # 偏移 1，对应 pad 后的实际位置
                grid_theta = (t_idx / (nθ_lat + 2 - 1)) * 2.0 - 1.0
            else:
                grid_theta = torch.linspace(-1, 1, out_n_theta, device=device)

            # ==========================================================
            # 2. 修正物理逻辑: R 轴使用局部面积计算插值权重，杜绝空间扭曲
            # ==========================================================
            r_out_val = torch.linspace(0, 1, out_n_R, device=device)
            
            # 找到目标落在原图哪两个像素之间 (0 到 nR_lat-2)
            float_idx = r_out_val * (nR_lat - 1)
            i = torch.floor(float_idx).clamp(0, nR_lat - 2)
            
            r0 = i / (nR_lat - 1)
            r1 = (i + 1) / (nR_lat - 1)
            
            # ✨ 计算真实的物理面积比例 (局部位移)
            w_r = (r_out_val**2 - r0**2) / (r1**2 - r0**2 + 1e-8)
            w_r = w_r.clamp(0.0, 1.0)
            
            # 伪造坐标: Base Index + 面积权重
            y_idx = i + w_r
            grid_R = (y_idx / (nR_lat - 1)) * 2.0 - 1.0

            # 生成 3D 网格 (Z, Theta, R) -> (D, H, W)
            mesh_Z, mesh_theta, mesh_R = torch.meshgrid(grid_Z, grid_theta, grid_R, indexing='ij')
            
            # 形状: (1, D, H, W, 3) 对应 grid_sample 需要的 (x=R, y=Theta, z=Z)
            grid_3d = torch.stack([mesh_R, mesh_theta, mesh_Z], dim=-1).unsqueeze(0)
            
            # 更新缓存
            self.grid_3d = grid_3d
            self.cached_in_shape = x.shape[2:]
            self.cached_out_shape = target_size

        # ==========================================================
        # 3. 前向传播：执行 Padding 与 采样
        # ==========================================================
        if scale_theta > 1:
            x = F.pad(x, pad=(0, 0, 1, 1, 0, 0), mode='circular')
            
        # 扩展 grid 到 Batch Size
        grid = self.grid_3d.expand(B, -1, -1, -1, -1)
        
        # 使用 align_corners=True (因为我们是基于 idx 计算的严谨网格)
        x_up = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        # 🎊 不需要再 slice 切割输出了！速度起飞！
        return x_up

class AttentionBlock3D(nn.Module):
    """
    搭载 PyTorch 2.0 SDPA (FlashAttention) 的 3D 空间多头注意力块。
    支持显存极速优化，严格物理边界屏蔽。
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        # 确保通道数能被头数整除
        assert channels % num_heads == 0, "channels 必须能被 num_heads 整除"
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        
        # 依然使用 1x1x1 Conv3d 提取特征，保持 3D 空间的不变性
        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, C, D, H, W = x.shape
        N = D * H * W
        
        # 1. 归一化
        h = self.norm(x)
        
        # 2. 计算 QKV (此时依然是 5D 张量)
        qkv = self.qkv(h) # (B, 3C, D, H, W)
        
        # 3. 形状变换，准备多头注意力
        # (B, 3C, D, H, W) -> (B, 3C, N) -> (B, N, 3C)
        qkv = qkv.view(B, 3 * C, N).permute(0, 2, 1).contiguous()
        
        # 切分为 Q, K, V，各形状: (B, N, C)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # 拆分为多头: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 4. 构建安全的 Attention Mask
        if mask is not None:
            # 🌟 绝杀：只提供 Key Mask，形状 (B, 1, 1, N)
            # PyTorch SDPA 会自动将其广播为 (B, num_heads, N_query, N_key)
            # True 表示有效，False 表示屏蔽
            attn_mask = mask.view(B, 1, 1, N).bool()
        else:
            attn_mask = None
            
        # 5. 调用底层优化的 SDPA (FlashAttention / Memory-Efficient Attention)
        # 它内部会自动处理 softmax(Q*K^T/sqrt(d)) * V
        out = F.scaled_dot_product_attention(
            query=q, 
            key=k, 
            value=v, 
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        # 6. 还原回 3D 物理场形状
        # (B, num_heads, N, head_dim) -> (B, N, num_heads, head_dim) -> (B, N, C)
        out = out.transpose(1, 2).reshape(B, N, C)
        
        # (B, N, C) -> (B, C, N) -> (B, C, D, H, W)
        out = out.permute(0, 2, 1).view(B, C, D, H, W).contiguous()
        
        # 7. 线性投影与最终的 Query 物理截断
        out = self.proj_out(out)
        
        if mask is not None:
            # 在这里，让石头 Query 瞎算出来的垃圾特征飞灰湮灭！
            out = out * mask
            
        return x + out

class ResNetBlock3D(nn.Module):
    """
    3D 残差块：在不引起梯度消失的前提下，成倍增加非线性表达能力
    """
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = Conv3dCylindrical(channels, channels, kernel_size=3, stride=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = Conv3dCylindrical(channels, channels, kernel_size=3, stride=1)
        
    def forward(self, x, mask=None):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        if mask is not None: 
            h = h * mask
        
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        if mask is not None: 
            h = h * mask
            
        return x + h


class AutoPhysicsScaler(nn.Module):
    """
    基础物理量纲缩放器。
    对输入 5D 物理场进行自适应仿射变换，将其映射至 N(0, 1) 附近。
    """
    def __init__(self, channels):
        super().__init__()
        # 统一由基类接管 Parameter 的注册
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
        self.shift = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class VQEncoderPhysicsScaler(AutoPhysicsScaler):
    """
    Encoder 专用缩放器。全通道执行仿射变换。
    """
    def __init__(self, channels):
        super().__init__(channels)
        # forward 方法直接继承父类，无需重写


class VQDecoderPhysicsScaler(AutoPhysicsScaler):
    """
    Decoder 专用缩放器。
    功能 1：对不受限物理量 (P, Ψ) 进行逆向仿射变换。
    功能 2：对组分饱和度 (Cw/Sw, Co, Cg) 执行通道 Softmax 硬约束，保证和为 1。
    """
    def __init__(self, channels, sat_indices=[1, 2, 3]):
        self.channels = channels
        self.sat_indices = sat_indices
        self.non_sat_indices = [i for i in range(channels) if i not in sat_indices]
        
        # 🌟 绝妙之处：把非饱和度通道的数量传给父类，让父类去建立对应形状的 scale 和 shift
        super().__init__(len(self.non_sat_indices))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        
        # 1. 对饱和度通道执行 Channel Softmax (强制 Sw + So + Sg = 1)
        if self.sat_indices:
            S_logits = x[:, self.sat_indices, ...]
            S_probs = F.softmax(S_logits, dim=1)
            out[:, self.sat_indices, ...] = S_probs
        
        # 2. 对其他物理量(P, Ψ)执行逆向仿射变换
        if self.non_sat_indices:
            non_sat_x = x[:, self.non_sat_indices, ...]
            # 🌟 优雅调用：直接利用父类的 forward 来做仿射变换
            out[:, self.non_sat_indices, ...] = super().forward(non_sat_x)
        
        return out

class MinMaxScaler(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0 * (x - self.min_val) / (self.max_val - self.min_val) - 1.0

    def unscale(self, x_scaled: torch.Tensor) -> torch.Tensor:
        x_unscaled = (x_scaled + 1.0) / 2.0 * (self.max_val - self.min_val) + self.min_val
        return torch.clamp(x_unscaled, min=self.min_val, max=self.max_val)

class LogScaler(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.log_min = torch.log(torch.tensor(min_val, dtype=torch.float32))
        self.log_max = torch.log(torch.tensor(max_val, dtype=torch.float32))

    def scale(self, x: torch.Tensor) -> torch.Tensor:
        x_safe = torch.clamp(x, min=self.min_val)
        x_log = torch.log(x_safe)
        return 2.0 * (x_log - self.log_min.to(x.device)) / (self.log_max.to(x.device) - self.log_min.to(x.device)) - 1.0

    def unscale(self, x_scaled: torch.Tensor) -> torch.Tensor:
        x_log = (x_scaled + 1.0) / 2.0 * (self.log_max.to(x_scaled.device) - self.log_min.to(x_scaled.device)) + self.log_min.to(x_scaled.device)
        x_unscaled = torch.exp(x_log)
        return torch.clamp(x_unscaled, min=self.min_val)


