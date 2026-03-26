import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# 导入动态数据集和模型
from vae.vq.dataset_vq_vae import DatasetVQVAE3D
from vae.vq.vq_vae import VQVAE3D

def setup_logger(log_dir: Path, prefix: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{prefix}.log"
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] - %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])
    return logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE (Dynamic Fields Stage 1)")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. 绝对路径与统一命名前缀 (时间_场名1_场名2)
    current_path = Path(__file__).resolve().parent
    current_time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    log_dir, ckpt_dir = current_path / "log", current_path / "checkpoints"
    
    # 拼接场名，例如 "pressure_swat"
    field_str = "_".join(cfg['field_names']).lower()
    prefix = f"{current_time}_{field_str}"
    
    logger = setup_logger(log_dir, prefix)
    logger.info(f"🚀 启动 Stage 1 VQ-VAE 训练 | 通道: {field_str}")

    # 2. 准备 Dataset 与 DataLoader (极简流式发牌，无需预扫极值)
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

    logger.info("📦 正在初始化 Train/Val 视图 (零开销)...")
    train_dataset = DatasetVQVAE3D(mode="train", shuffle=True, **ds_params)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=0)

    val_dataset = DatasetVQVAE3D(mode="val", shuffle=False, **ds_params)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=0)

    # 3. 实例化 VQ-VAE 模型
    model = VQVAE3D(
        in_channels=cfg['model']['in_channels'],
        hidden_dim=cfg['model']['hidden_dim'],
        latent_dim=cfg['model']['latent_dim'],
        num_embeddings=cfg['model']['num_embeddings'],
        commitment_cost=cfg['model']['commitment_cost'],
        num_res_blocks=cfg['model']['num_res_blocks'],
        num_attn_heads=cfg['model']['num_attn_heads'],
        compress_shape=tuple(cfg['model']['compress_shape'])
    )

    # 4. Logger & 极致省空间的 Checkpoint
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=f"{prefix}_tb")
    csv_logger = CSVLogger(save_dir=log_dir, name=f"{prefix}_csv")
    
    # 自动覆盖，始终只保留 1 个最优模型
    ckpt_filename = f"{prefix}_best"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_filename,
        save_top_k=1,                     
        monitor="val_loss",
        mode="min",
        enable_version_counter=False      # 防止产生 -v1, -v2 后缀
    )

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        logger=[tb_logger, csv_logger],
        log_every_n_steps=5
    )

    logger.info("🔥 开始训练 VQ-VAE...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("✅ 训练结束!")

if __name__ == "__main__":
    main()