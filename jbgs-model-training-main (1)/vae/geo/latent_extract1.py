import torch
import torch.nn.functional as F
from geo_vae import PoroVAE3D, PermVAE3D


def downsample_static_latents(z_poro, z_perm, latent_mask, target_hw=(10, 10)):
    """
    将静态 latent 从 (B,C,20,20,5) 降到 (B,C,10,10,5)
    只对前两个空间维做下采样，最后一个维度 R 保持不变。
    """
    if z_poro.dim() != 5 or z_perm.dim() != 5 or latent_mask.dim() != 5:
        raise ValueError("z_poro, z_perm, latent_mask 都必须是 5 维张量")

    if z_poro.shape[2:] != z_perm.shape[2:]:
        raise ValueError(f"z_poro 空间尺寸 {z_poro.shape[2:]} 与 z_perm 空间尺寸 {z_perm.shape[2:]} 不一致")

    if z_poro.shape[2:] != latent_mask.shape[2:]:
        raise ValueError(f"latent_mask 空间尺寸 {latent_mask.shape[2:]} 与 latent 不一致")

    B, C1, H, W, R = z_poro.shape
    _, C2, _, _, _ = z_perm.shape
    th, tw = target_hw

    # (B, C, H, W, R) -> (B*R, C, H, W)
    z_poro_2d = z_poro.permute(0, 4, 1, 2, 3).contiguous().reshape(B * R, C1, H, W)
    z_perm_2d = z_perm.permute(0, 4, 1, 2, 3).contiguous().reshape(B * R, C2, H, W)
    mask_2d = latent_mask.permute(0, 4, 1, 2, 3).contiguous().reshape(B * R, 1, H, W).float()

    # 特征：平均池化
    z_poro_ds_2d = F.adaptive_avg_pool2d(z_poro_2d, output_size=(th, tw))
    z_perm_ds_2d = F.adaptive_avg_pool2d(z_perm_2d, output_size=(th, tw))

    # mask：最近邻下采样
    mask_ds_2d = F.interpolate(mask_2d, size=(th, tw), mode="nearest")
    mask_ds_2d = (mask_ds_2d > 0.5).float()

    # 还原回 5D: (B*R, C, th, tw) -> (B, C, th, tw, R)
    z_poro_ds = z_poro_ds_2d.reshape(B, R, C1, th, tw).permute(0, 2, 3, 4, 1).contiguous()
    z_perm_ds = z_perm_ds_2d.reshape(B, R, C2, th, tw).permute(0, 2, 3, 4, 1).contiguous()
    mask_ds = mask_ds_2d.reshape(B, R, 1, th, tw).permute(0, 2, 3, 4, 1).contiguous()

    return z_poro_ds, z_perm_ds, mask_ds


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

poro_ckpt = r"E:\jbgs-model-training-main(5)\vae\geo\checkpoints\2026-03-15_00.48.34_poro_best.ckpt"
perm_ckpt = r"E:\jbgs-model-training-main(5)\vae\geo\checkpoints\2026-03-14_21.43.03_permx_best.ckpt"

poro_vae = PoroVAE3D.load_from_checkpoint(
    poro_ckpt,
    map_location=device,
    poro_min=0.0,
    poro_max=0.4,
    strict=False,
).to(device)

perm_vae = PermVAE3D.load_from_checkpoint(
    perm_ckpt,
    map_location=device,
    perm_min=0.1,
    perm_max=10000.0,
    strict=False,
).to(device)

poro_vae.eval()
perm_vae.eval()

print("加载成功")

# 这里先用示例输入；后面你可以换成真实 poro / perm / mask
B, nz, n_theta, n_R = 2, 40, 40, 10
poro_field = torch.rand(B, 1, nz, n_theta, n_R, device=device)
perm_field = torch.rand(B, 1, nz, n_theta, n_R, device=device)
spatial_mask = torch.ones(B, 1, nz, n_theta, n_R, device=device)

# 1. 提原始潜特征
with torch.no_grad():
    z_poro, latent_mask_poro = poro_vae.encode_latent(
        poro_field, spatial_mask, use_mean=True
    )
    z_perm, latent_mask_perm = perm_vae.encode_latent(
        perm_field, spatial_mask, use_mean=True
    )

print("原始 latent:")
print("z_poro shape:", z_poro.shape)
print("latent_mask_poro shape:", latent_mask_poro.shape)
print("z_perm shape:", z_perm.shape)
print("latent_mask_perm shape:", latent_mask_perm.shape)

# 2. 保存原始 latent
torch.save(
    {
        "z_poro": z_poro.cpu(),
        "latent_mask_poro": latent_mask_poro.cpu(),
        "z_perm": z_perm.cpu(),
        "latent_mask_perm": latent_mask_perm.cpu(),
    },
    "static_latents.pt"
)
print("原始潜特征已保存到 static_latents.pt")

# 3. 降采样到 10x10x5
z_poro_ds, z_perm_ds, latent_mask_ds = downsample_static_latents(
    z_poro, z_perm, latent_mask_poro, target_hw=(10, 10)
)

print("降采样后 latent:")
print("z_poro_ds shape:", z_poro_ds.shape)
print("latent_mask_ds shape:", latent_mask_ds.shape)
print("z_perm_ds shape:", z_perm_ds.shape)

# 4. 保存降采样后的 latent
torch.save(
    {
        "z_poro": z_poro_ds.cpu(),
        "latent_mask_poro": latent_mask_ds.cpu(),
        "z_perm": z_perm_ds.cpu(),
        "latent_mask_perm": latent_mask_ds.cpu(),
    },
    "static_latents_10x10x5.pt"
)
print("降采样潜特征已保存到 static_latents_10x10x5.pt")