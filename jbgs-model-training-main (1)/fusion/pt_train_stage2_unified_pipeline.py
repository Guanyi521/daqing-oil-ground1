import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# ==========================================
# 引入项目内的模型与数据集组件
# ==========================================
# 1. 引入 Stage 1 的模型架构
from vae.vq.vq_vae import VQVAE3D
from vae.geo.geo_vae import PoroVAE3D, PermVAE3D

# 2. 引入 Stage 2 的核心组件
from fusion.models.transformer import DecoderFusionTransformer
from fusion.models.unified_pipeline import LatentDynamicsPipeline

# 3. 引入您编写的数据集
# 假设 dataset_unified_pipeline.py 放在 fusion/ 目录下
from dataset_unified_pipeline import DatasetUnifiedPipeline


def main():
    # ==========================================
    # 1. 解析命令行参数并加载 YAML 配置
    # ==========================================
    parser = argparse.ArgumentParser(description="RFM Stage 2: Transformer Fusion Training")
    parser.add_argument('--config', type=str, default='unified_pipeline_config.yaml', help='YAML 配置文件路径')
    cmd_args = parser.parse_args()

    print(f"📄 正在加载配置文件: {cmd_args.config}")
    if not os.path.exists(cmd_args.config):
        raise FileNotFoundError(f"❌ 找不到配置文件: {cmd_args.config}")

    with open(cmd_args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ==========================================
    # 2. 准备数据集 (Dataset & DataLoader)
    # ==========================================
    print("🚀 正在初始化 DatasetUnifiedPipeline...")

    train_dataset = DatasetUnifiedPipeline(
        project_dirs=config['data']['train_dirs'],
        project_names=config['data']['train_names'],
        dynamic_fields=config['data']['dynamic_fields'],
        nt_width=config['data']['nt_width'],
        stride=config['data'].get('stride', 1),
        mode="train",
        nums_loading_projects=config['training'].get('nums_loading_projects', 2),
        nθ=config['data']['n_theta'],
        nR=config['data']['n_R'],
        model_nz=config['data']['model_nz'],
        maxR=config['data']['maxR'],
        total_z_depth=config['data']['total_z_depth']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )

    # ==========================================
    # 3. 实例化 Stage 1 模型并加载预训练权重
    # ==========================================
    print("🧠 正在构建 Stage 1 视觉/地质神经 (VAEs) 并加载权重...")

    # 3.1 VQ-VAE (动态场)
    vqvae = VQVAE3D(**config['vq_vae_config'])
    vq_ckpt_path = config['stage1_checkpoints']['vq_vae_path']
    if os.path.exists(vq_ckpt_path):
        vq_ckpt = torch.load(vq_ckpt_path, map_location='cpu')
        vqvae.load_state_dict(vq_ckpt.get('state_dict', vq_ckpt), strict=False)
        print("  ✅ VQ-VAE 权重加载成功")
    else:
        print(f"  ⚠️ 警告: 未找到 VQ-VAE 权重 ({vq_ckpt_path})")

    # 3.2 Geo-VAE (静态孔隙度 Poro)
    poro_vae = PoroVAE3D(**config['geo_vae_poro_config'])
    poro_ckpt_path = config['stage1_checkpoints']['geo_vae_poro_path']
    if os.path.exists(poro_ckpt_path):
        poro_ckpt = torch.load(poro_ckpt_path, map_location='cpu')
        poro_vae.load_state_dict(poro_ckpt.get('state_dict', poro_ckpt), strict=False)
        print("  ✅ Poro-VAE 权重加载成功")

    # 3.3 Geo-VAE (静态渗透率 Perm)
    perm_vae = PermVAE3D(**config['geo_vae_perm_config'])
    perm_ckpt_path = config['stage1_checkpoints']['geo_vae_perm_path']
    if os.path.exists(perm_ckpt_path):
        perm_ckpt = torch.load(perm_ckpt_path, map_location='cpu')
        perm_vae.load_state_dict(perm_ckpt.get('state_dict', perm_ckpt), strict=False)
        print("  ✅ Perm-VAE 权重加载成功")

    # ==========================================
    # 4. 实例化 Stage 2 模型并组装 Pipeline
    # ==========================================
    print("⚙️ 正在构建 Transformer 推演引擎...")
    transformer = DecoderFusionTransformer(**config['transformer_config'])

    print("🔗 正在组装 LatentDynamicsPipeline...")
    # 完全匹配您 unified_pipeline.py 中的 __init__ 接口
    model = LatentDynamicsPipeline(
        vqvae_model=vqvae,
        transformer_model=transformer,
        poro_vae=poro_vae,
        perm_vae=perm_vae,
        d_model=config['transformer_config']['d_model'],
        freeze_vqvae=config['model'].get('freeze_vqvae', True),
        freeze_static_vae=config['model'].get('freeze_poro_vae', True),
        dummy_dynamic_groups=1
    )

    # ==========================================
    # 5. 配置 PyTorch Lightning Trainer
    # ==========================================
    # 日志记录器
    logger = TensorBoardLogger("tb_logs", name="stage2_fusion")

    # 断点保存策略：保存最新的和 Loss 最好的
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/fusion/",
        filename="fusion-{epoch:02d}-{train_pipeline_loss:.4f}",
        save_top_k=3,
        monitor="train_pipeline_loss",  # 与 unified_pipeline 中 self.log('train_pipeline_loss', ...) 对应
        mode="min",
        every_n_epochs=1,
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=-1 if torch.cuda.is_available() else 1,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",  # 自动多卡并行
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="16-mixed",  # 开启混合精度，大幅加速 Transformer 并节省显存
        log_every_n_steps=10
    )

    # ==========================================
    # 6. 启动训练！
    # ==========================================
    print("🔥 万事俱备，开始 Stage 2 训练...")
    trainer.fit(model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()