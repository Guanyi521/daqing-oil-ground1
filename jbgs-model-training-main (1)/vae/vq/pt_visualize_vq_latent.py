import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 复用您写好的现成模块
from vae.vq.vq_vae import VQVAE3D
from vae.vq.dataset_vq_vae import DatasetVQVAE3D
from torch.utils.data import DataLoader

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Visualize VQ-VAE Latent Space")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained .ckpt file")
    parser.add_argument("--max_samples", type=int, default=20000, help="Max spatial points to plot")
    args = parser.parse_args()

    # 1. 加载配置
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ 当前运行设备: {device}")

    # 2. 自动恢复 Lightning 模型权重
    print(f"📦 正在加载模型权重: {args.ckpt}")
    model = VQVAE3D.load_from_checkpoint(args.ckpt)
    model.eval()
    model.to(device)

    # 3. 初始化 DataLoader 以获取真实的动态场数据
    print("🌊 正在加载真实的动态场地质数据...")
    ds_params = {
        "project_dirs": cfg['project_dirs'],
        "project_names": cfg['project_names'],
        "field_names": cfg['field_names'],
        "nums_loading_projects": cfg['nums_loading_projects'],
        "model_nz": cfg['model_nz'],
        "nθ": cfg['n_theta'],
        "nR": cfg['n_R'],
        "train_ratio": cfg['train_val_split'],
        "seed": cfg['seed']
    }
    
    # 建议用 val 验证集数据来观测特征分布
    val_dataset = DatasetVQVAE3D(mode="val", shuffle=True, **ds_params)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=0)

    # =========================================================
    # 🌟 核心提取 1: 获取真实的连续特征 z_e
    # =========================================================
    print("🔍 正在编码提取高维潜空间连续特征 (z_e)...")
    z_e_list = []
    for batch_idx, batch in enumerate(val_loader):
        x, hard_data_locs, mask = batch
        x, mask = x.to(device), mask.to(device)
        
        # 调用现成的 Encoder，获取 mu (推理时直接用 mu，不加噪声)
        mu, _, latent_mask = model.encoder(x, mask)
        
        # 维度转换: (B, latent_dim, nz, n_theta, n_R) -> (N, latent_dim)
        B, C, nz, n_theta, n_R = mu.shape
        mu_permuted = mu.permute(0, 2, 3, 4, 1).reshape(-1, C)
        mask_flat = latent_mask.permute(0, 2, 3, 4, 1).reshape(-1)
        
        # 过滤掉石头(无效网格)的数值
        valid_mu = mu_permuted[mask_flat > 0.5]
        
        # 🚨 极其重要：与您的 Spherical VQ 层对齐，必须投射到超球面上！
        valid_mu_norm = F.normalize(valid_mu, p=2, dim=1)
        z_e_list.append(valid_mu_norm.cpu().numpy())
        
        # 凑够指定数量的点就停止，避免 PCA 内存爆炸
        if sum(len(v) for v in z_e_list) >= args.max_samples:
            break

    z_e_continuous = np.concatenate(z_e_list, axis=0)[:args.max_samples]
    print(f"✅ 成功提取 {z_e_continuous.shape[0]} 个有效网格的物理连续特征。")

    # =========================================================
    # 🌟 核心提取 2: 获取学习到的离散码本 z_q
    # =========================================================
    print("📖 正在提取 Codebook 密码本锚点 (z_q)...")
    # 绕过 index=0 的 Padding，提取真实的 512/1024 个锚点
    raw_weights = model.vq_layer.embedding.weight[1:] 
    
    # 🚨 极其重要：与您的 Spherical VQ 对齐，码本必须处于同一超球面上！
    z_q_norm = F.normalize(raw_weights, p=2, dim=1).cpu().numpy()
    print(f"✅ 成功提取 {z_q_norm.shape[0]} 个离散物理字典 Token。")

    # =========================================================
    # 🌟 降维打击与绘图 (PCA / t-SNE)
    # =========================================================
    print("🌌 正在进行主成分降维并渲染图表...")
    pca = PCA(n_components=2)
    # 以海量的真实数据作为主干基底进行拟合
    z_e_2d = pca.fit_transform(z_e_continuous)
    # 将码本锚点投影到同一个基底的平面上
    z_q_2d = pca.transform(z_q_norm)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 1. 绘制动态场的物理连续特征密度图 (星云)
    sns.kdeplot(
        x=z_e_2d[:, 0], y=z_e_2d[:, 1], 
        cmap="Blues", fill=True, thresh=0.05, levels=15, alpha=0.8, ax=ax
    )

    # 2. 绘制离散码本锚点 (恒星)
    ax.scatter(
        z_q_2d[:, 0], z_q_2d[:, 1], 
        c="#FF3300", s=30, edgecolor="white", linewidth=0.8, alpha=0.9,
        marker="o", label=f"Codebook Anchors ({z_q_norm.shape[0]} Tokens)"
    )

    # 3. 添加少许底层真实数据作为点缀，增加质感
    ax.scatter(
        z_e_2d[:1500, 0], z_e_2d[:1500, 1], 
        c="black", s=1, alpha=0.15, label="Continuous Latent Features ($z_e$)"
    )

    ax.set_title("Learned Latent Space: Real Data Density vs. Codebook Anchors", 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("PCA Dimension 1 (Spherical Latent Space)", fontsize=13)
    ax.set_ylabel("PCA Dimension 2 (Spherical Latent Space)", fontsize=13)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    leg = ax.legend(loc="upper right", fontsize=11, frameon=True, shadow=True, edgecolor='black')
    for lh in leg.legend_handles: 
        lh.set_alpha(1)

    save_path = "vq_latent_space_real.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"🎉 大功告成！真实数据图谱已保存至: {save_path}")

if __name__ == "__main__":
    main()