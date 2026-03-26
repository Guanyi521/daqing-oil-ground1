# 核心！Transformer 数据集：负责时间窗口的 Chunking 截断与重叠滑动逻辑

import os
import glob
import torch
from torch.utils.data import Dataset

class ReservoirTransformerDataset(Dataset):
    """
    第二阶段 Transformer 的滑动窗口自回归数据集。
    输入数据应为 Stage 1 (VQ-VAE & VAE-Geo) 预提取好的离散 Token 和连续特征。
    """
    def __init__(self, data_dir: str, nt_width: int, stride: int = 1):
        """
        Args:
            data_dir (str): 预提取 Token 数据目录。
            nt_width (int): 时间窗口的宽度 (Transformer 需要基于过去/现在的 nt_width 步，预测未来的状态)。
            stride (int): 滑动窗口的步长。设为 1 表示重叠最大，数据量最多。
        """
        super().__init__()
        self.nt_width = nt_width
        self.stride = stride
        
        # 查找所有预处理好的整条井组生命周期轨迹序列
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if len(self.file_paths) == 0:
            print(f"Warning: 在 {data_dir} 中未找到任何序列数据文件！")

        # 构建全局索引映射表 (Global Index Mapping)
        # 因为每条轨迹的长度 T 不同，我们需要建立一个 idx -> (file_path, start_t) 的映射
        self.samples = []
        for file_path in self.file_paths:
            # 轻量级加载 meta 信息（避免把所有数据爆内存）
            meta = torch.load(file_path, map_location='cpu')
            T = meta['tokens'].shape[0] # 时间序列总长度
            
            # 计算这条序列能切出多少个长度为 nt_width + 1 的窗口
            # (需要 nt_width 个作为输入，外加 1 个作为最后一个时间步的预测目标)
            max_start = T - (nt_width + 1)
            for start_t in range(0, max_start + 1, stride):
                self.samples.append((file_path, start_t))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, start_t = self.samples[idx]
        
        # 加载对应的轨迹数据
        data_dict = torch.load(file_path, map_location='cpu')
        
        # 1. 切片提取时间窗口 Token 序列 (nt_width + 1, Ld)
        end_t = start_t + self.nt_width + 1
        window_tokens = data_dict['tokens'][start_t : end_t].clone() 
        
        # 2. 提取静态多模态特征与掩码
        features = data_dict['features']           # 形状: (n_features, L_src, d_model)
        grid_shape = data_dict['grid_shape']       # 形状: tuple(nz_latent, n_theta_latent, n_R_latent)
        spatial_mask = data_dict['spatial_mask']   # 形状: (Ld,)
        feature_mask = data_dict['feature_mask']   # 形状: (L_src,)
        
        return window_tokens, features, grid_shape, spatial_mask, feature_mask


# =========================================================================
#  离线特征提取流水线 (Bridging Stage 1 and Stage 2)
# =========================================================================
@torch.no_grad()
def generate_transformer_offline_dataset(
    raw_sequence_dir: str, 
    save_dir: str, 
    vqvae_model, 
    geo_vae_model, 
    device='cuda'
):
    """
    离线提取流水线：
    读取完整的物理时间序列和地质场，使用冻结的 Stage 1 模型将其压扁成 Token 和 Feature。
    """
    os.makedirs(save_dir, exist_ok=True)
    vqvae_model.eval().to(device)
    geo_vae_model.eval().to(device)
    
    # 假设 raw_sequence_dir 中有完整的时序数据
    seq_files = sorted(glob.glob(os.path.join(raw_sequence_dir, "*.pt")))
    
    for idx, file_path in enumerate(seq_files):
        # 假设加载的原始数据形状:
        # dynamic_field: (T, 3, nz, n_theta, n_R)
        # static_geo: (2, nz, n_theta, n_R)
        # well_controls: (n_well_features, L_well_src, d_model) -> 比如注入量/流压的显示映射
        # mask: (1, nz, n_theta, n_R)
        raw_data = torch.load(file_path)
        dynamic_field = raw_data['dynamic_field'].to(device)
        static_geo = raw_data['static_geo'].unsqueeze(0).to(device) # (1, 2, nz, n_theta, n_R)
        well_controls = raw_data['well_controls'].to(device) 
        orig_mask = raw_data['mask'].unsqueeze(0).to(device) # (1, 1, nz, n_theta, n_R)
        
        T = dynamic_field.shape[0]
        
        # 1. 生成潜空间掩码 (Ld,)
        import torch.nn.functional as F
        # 假设 VQ-VAE 下采样使得特征尺度缩减了对应的倍数，这里简写模拟下采样
        latent_mask = F.interpolate(orig_mask.float(), scale_factor=0.5, mode='nearest').bool()
        grid_shape = latent_mask.shape[2:] # (nz_latent, n_theta_latent, n_R_latent)
        spatial_mask = latent_mask.view(-1) # 展平为 (Ld,)
        
        # 2. 批量提取动态 Token IDs (分批防止 OOM)
        batch_size = 16 
        all_tokens = []
        for i in range(0, T, batch_size):
            batch_field = dynamic_field[i:i+batch_size]
            # 扩展 mask 以匹配 batch size
            batch_mask = latent_mask.expand(batch_field.shape[0], -1, -1, -1, -1)
            
            # 调用我们在 vqvae.py 中写好的纯净接口
            indices, _, _ = vqvae_model.encode_to_indices(batch_field, batch_mask)
            
            # 展平空间维度: (B, nz', n_theta', n_R') -> (B, Ld)
            tokens_flat = indices.view(batch_field.shape[0], -1)
            all_tokens.append(tokens_flat.cpu())
            
        all_tokens = torch.cat(all_tokens, dim=0) # 形状: (T, Ld)
        
        # 3. 提取静态地质全局 Feature (1, d_model) -> 扩展为 (1, 1, d_model)
        # 在多特征融合架构里，需要保证特征形状统一为 (L_src, d_model)
        geo_feature = geo_vae_model.get_geo_prompt(static_geo) # (1, 1, d_model)
        
        # 假设 well_controls 已经是 (1, L_well_src, d_model) 的形状
        # 4. 组装多模态 Features: (1, 2, L_max_src, d_model)
        # 实际工程中可能由于 L_src 不一致，需要用 feature_mask 进行 Padding 处理
        # 这里用简化的伪代码合并展示
        features = torch.stack([
            geo_feature.expand(1, well_controls.shape[1], -1), # 广播补齐长度
            well_controls
        ], dim=1).cpu()
        
        # 对于这个例子，假设没有 Padding，全部有效
        feature_mask = torch.ones(features.shape[2], dtype=torch.bool)
        
        # 5. 保存极速训练就绪的数据包
        processed_data = {
            'tokens': all_tokens,                 # (T, Ld)
            'features': features.squeeze(0),      # (2, L_src, d_model)
            'grid_shape': grid_shape,             # tuple
            'spatial_mask': spatial_mask.cpu(),   # (Ld,)
            'feature_mask': feature_mask          # (L_src,)
        }
        
        torch.save(processed_data, os.path.join(save_dir, f"seq_{idx:04d}.pt"))
        
    print("Stage 2 Transformer Dataset Generation Complete!")