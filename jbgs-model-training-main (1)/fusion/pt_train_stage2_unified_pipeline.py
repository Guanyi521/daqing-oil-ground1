# 第二阶段：冻结 Stage1 模型，在潜空间中训练 Transformer 自回归

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from omegaconf import OmegaConf  # 引入 OmegaConf

# 导入核心 Transformer 模型与离线数据集
from models.transformer import DecoderFusionTransformer
from fusion.dataset_unified_pipeline import ReservoirTransformerDataset

def train_transformer(cfg):
    print("🚀 正在启动 RFM Stage 2: Transformer 时空多模态自回归预训练...")
    
    # =========================================================================
    # 1. 数据集加载与滑动窗口切分 (从 cfg.data 读取)
    # =========================================================================
    dataset = ReservoirTransformerDataset(
        data_dir=cfg.data.data_dir, 
        nt_width=cfg.data.nt_width, 
        stride=cfg.data.stride
    )
    
    if len(dataset) == 0:
        raise ValueError(f"❌ 在 {cfg.data.data_dir} 中未找到序列数据！请先运行离线特征提取流水线。")
        
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 极速 DataLoader 配置 (从 cfg.training 读取)
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, 
                              num_workers=cfg.training.num_workers, pin_memory=True)

    print(f"✅ 数据加载成功！共生成 {len(dataset)} 个滑动窗口样本。")

    # =========================================================================
    # 2. 初始化 Decoder-only Transformer 骨干网络 (从 cfg.model 读取)
    # =========================================================================
    model = DecoderFusionTransformer(
        vocab_size=cfg.model.vocab_size, 
        d_model=cfg.model.d_model, 
        num_heads=cfg.model.num_heads, 
        # OmegaConf 解析出来的是 ListConfig，转为普通 list 以策安全
        head_splits=list(cfg.model.head_splits), 
        num_layers=cfg.model.num_layers, 
        hidden_dim=cfg.model.hidden_dim,
        max_nt_width=cfg.model.max_nt_width, 
        Ld=cfg.data.Ld                      
    )

    # =========================================================================
    # 3. 设置训练回调 (Callbacks & Logger)
    # =========================================================================
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.training.save_dir, "transformer_checkpoints"),
        filename="rfm-llm-{epoch:03d}-{train_loss:.4f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        every_n_train_steps=1000 # LLM 训练通常在 epoch 中途也需要存盘
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(cfg.training.log_dir, name="transformer_logs")

    # =========================================================================
    # 4. 启动大模型专用的极速 Trainer (从 cfg.training 读取)
    # =========================================================================
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu",
        devices=cfg.training.devices,
        strategy="ddp" if cfg.training.devices > 1 else "auto", 
        precision="16-mixed" if cfg.training.use_amp else "32-true", 
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0, 
        accumulate_grad_batches=cfg.training.accumulate_grad_batches 
    )

    trainer.fit(model, train_loader)
    print("🎉 Transformer 自回归基础模型 (Stage 2) 训练完成！")

# =========================================================================
# 命令行解析与主控入口
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFM Stage 2: Training Autoregressive Transformer")
    
    # 唯一保留的命令行参数：配置文件路径
    parser.add_argument('--config', type=str, default='configs/transformer_config.yaml', help='YAML 配置文件路径')
    cmd_args = parser.parse_args()

    # 1. 极其优雅地加载配置
    cfg = OmegaConf.load(cmd_args.config)
    
    # 2. 打印配置字典，用于实验跟踪和日志记录
    print("\n" + "="*50)
    print("🚀 EXPERIMENT CONFIGURATION (STAGE 2)")
    print("="*50)
    print(OmegaConf.to_yaml(cfg))
    print("="*50 + "\n")

    # 开启 PyTorch 2.0+ Tensor Core 极速矩阵乘法优化
    torch.set_float32_matmul_precision('medium') 

    train_transformer(cfg)