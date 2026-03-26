# 3D 柱状相对位置编码、自定义块状因果掩码、井控 CNN Patch Extractor
import torch
import math

def generate_euclidean_positional_encoding(
    grid_shape: tuple, 
    d_model: int, 
    maxR: float,          # 物理超参数: 例如 1000.0 (米)
    total_z_depth: float, # 物理超参数: 例如 50.0 (米)
    device: torch.device
) -> torch.Tensor:
    
    nz, n_theta, n_R = grid_shape
    L_tgt = nz * n_theta * n_R
    
    # 均分 d_model 给 x, y, z 三个正交维度
    d_x = (d_model // 3) // 2 * 2
    d_y = (d_model // 3) // 2 * 2
    d_z_emb = d_model - d_x - d_y
    
    # 1. 生成离散索引
    z_idx, theta_idx, R_idx = torch.meshgrid(
        torch.arange(nz, device=device),
        torch.arange(n_theta, device=device),
        torch.arange(n_R, device=device),
        indexing='ij'
    )
    
    # 2. 🌟 绝对物理坐标映射 (Physical Coordinate Mapping)
    # Z 轴物理坐标: [0, total_z_depth]
    z_phys = (z_idx.float() / max(1, nz - 1)) * total_z_depth
    
    # R 轴物理坐标: 考虑到你之前的面积加权 (r^2 映射)，我们用 sqrt() 保证外圈网格体积合理
    r_phys = torch.sqrt(R_idx.float() / max(1, n_R - 1)) * maxR
    
    # Theta 物理角度: [0, 2pi)
    theta_phys = (theta_idx.float() / n_theta) * 2 * math.pi
    
    # 🌟 柱坐标向笛卡尔欧氏空间投影
    x_phys = r_phys * torch.cos(theta_phys)
    y_phys = r_phys * torch.sin(theta_phys)
    
    # 展平为 (L_tgt,)
    x_phys = x_phys.flatten()
    y_phys = y_phys.flatten()
    z_phys = z_phys.flatten()

    # 3. 计算正弦编码 (频率基数 base 可以根据你最大物理距离 1000m 适当调大，比如 10000 -> 100000)
    base = 100000.0 
    
    # --- X 轴 ---
    pe_x = torch.zeros((L_tgt, d_x), device=device)
    div_term_x = torch.exp(torch.arange(0, d_x, 2, device=device).float() * (-math.log(base) / d_x))
    pe_x[:, 0::2] = torch.sin(x_phys.unsqueeze(1) * div_term_x)
    pe_x[:, 1::2] = torch.cos(x_phys.unsqueeze(1) * div_term_x)

    # --- Y 轴 ---
    pe_y = torch.zeros((L_tgt, d_y), device=device)
    div_term_y = torch.exp(torch.arange(0, d_y, 2, device=device).float() * (-math.log(base) / d_y))
    pe_y[:, 0::2] = torch.sin(y_phys.unsqueeze(1) * div_term_y)
    pe_y[:, 1::2] = torch.cos(y_phys.unsqueeze(1) * div_term_y)

    # --- Z 轴 ---
    pe_z = torch.zeros((L_tgt, d_z_emb), device=device)
    div_term_z = torch.exp(torch.arange(0, d_z_emb, 2, device=device).float() * (-math.log(base) / d_z_emb))
    pe_z[:, 0::2] = torch.sin(z_phys.unsqueeze(1) * div_term_z)
    pe_z[:, 1::2] = torch.cos(z_phys.unsqueeze(1) * div_term_z)

    # 4. 正交拼接
    pos_encoding = torch.cat([pe_x, pe_y, pe_z], dim=1)
    
    return pos_encoding


def generate_time_positional_encoding(nt_width: int, d_model: int, device: torch.device) -> torch.Tensor:
    """生成标准 1D 正弦时间位置编码。形状: (nt_width, d_model)"""
    pe = torch.zeros(nt_width, d_model, device=device)
    position = torch.arange(0, nt_width, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def generate_block_causal_mask(nt_width: int, Ld: int, device: torch.device) -> torch.Tensor:
    """
    基于张量广播的极速块状因果掩码生成
    形状: (N, N)，其中 N = nt_width * Ld
    """
    # 1. 生成 (nt, nt) 的标准上三角因果掩码 (True 表示不可见/遮蔽)
    # diagonal=1 表示主对角线全为 False (自己能看见自己，且当前时刻能看到当前时刻)
    mask_nt = torch.triu(torch.ones(nt_width, nt_width, dtype=torch.bool, device=device), diagonal=1)
    
    # 2. 魔法广播: (nt, nt) -> (nt, 1, nt, 1) -> (nt, Ld, nt, Ld)
    mask_expanded = mask_nt.unsqueeze(1).unsqueeze(3).expand(nt_width, Ld, nt_width, Ld)
    
    # 3. 展平为 (nt * Ld, nt * Ld)
    block_causal_mask = mask_expanded.reshape(nt_width * Ld, nt_width * Ld)
    
    return block_causal_mask # True 表示遮蔽(不可见)，False 表示可见