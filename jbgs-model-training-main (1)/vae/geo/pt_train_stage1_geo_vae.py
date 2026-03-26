import yaml
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from vae.geo.dataset_geo_vae import DatasetGeoVAE3D
from vae.geo.geo_vae import PoroVAE3D, PermVAE3D
from eclipse.model_data import EclipseModelData
from eclipse.well_data import EclipseWellData

def setup_logger(log_dir: Path, field_name: str, current_time: str):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{current_time}_{field_name.lower()}.log"
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] - %(message)s",
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler()])
    return logging.getLogger(__name__)

def pre_scan_and_split(cfg, logger):
    """全局扫描极值，并执行按井的 Train/Val 完美物理切分"""
    g_min, g_max = float('inf'), float('-inf')
    total_grids, total_wells = 0, 0
    
    train_dirs, train_names, train_wells = [], [], []
    val_dirs, val_names, val_wells = [], [], []
    
    split_ratio = cfg.get('train_val_split', 0.9)
    field_name = cfg['field_name']

    for p_dir, p_name in zip(cfg['project_dirs'], cfg['project_names']):
        try:
            model = EclipseModelData(p_dir, p_name)
            well_data = EclipseWellData(str(Path(p_dir) / f"{p_name}.DATA"))
            
            # 1. 极值扫描
            field_raw = model.get_static_field(field_name)
            valid_data = field_raw[field_raw > 1e-5] if field_name == "PERMX" else field_raw
            if len(valid_data) > 0:
                g_min = min(g_min, float(valid_data.min()))
                g_max = max(g_max, float(valid_data.max()))
                
            # 2. 规模统计与按井切分
            total_grids += model.base_active_cells
            last_step_df = well_data.get_steps_info(-1)
            wells_in_proj = last_step_df['well'].unique().tolist() if not last_step_df.empty else []
            total_wells += len(wells_in_proj)
            
            if len(wells_in_proj) > 0:
                random.shuffle(wells_in_proj)
                t_idx = max(1, int(len(wells_in_proj) * split_ratio))
                
                train_dirs.append(p_dir); train_names.append(p_name); train_wells.append(wells_in_proj[:t_idx])
                
                # 如果有剩余的井，划入验证集
                if t_idx < len(wells_in_proj):
                    val_dirs.append(p_dir); val_names.append(p_name); val_wells.append(wells_in_proj[t_idx:])
                    
        except Exception as e:
            logger.warning(f"扫描跳过项目 {p_name}: {e}")

    # 保底机制
    if g_min == float('inf'):
        g_min, g_max = (0.1, 10000.0) if field_name == "PERMX" else (0.0, 0.4)

    logger.info(f"📊 扫描完毕! 总有效网格: {total_grids:,} | 井数: {total_wells} | {field_name} 极值: [{g_min:.4f}, {g_max:.4f}]")
    
    return (g_min, g_max), (train_dirs, train_names, train_wells), (val_dirs, val_names, val_wells)

def main():
    parser = argparse.ArgumentParser(description="Train Geo VAE (Stage 1)")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # 1. 绝对路径与日志初始化 (支持 python -m)
    current_path = Path(__file__).resolve().parent
    current_time = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    log_dir, ckpt_dir = current_path / "log", current_path / "checkpoints"
    logger = setup_logger(log_dir, cfg['field_name'], current_time)

    logger.info(f"🚀 启动 Stage 1 geo-vae 训练: {cfg['field_name']}")

    # 2. 扫描数据与构建数据集
    (g_min, g_max), train_meta, val_meta = pre_scan_and_split(cfg, logger)

    ds_params = {
        "field_name": cfg['field_name'], "batch_size": cfg['batch_size'],
        "nums_loading_projects": cfg['nums_loading_projects'],
        "model_nz": cfg['model_nz'], "nθ": cfg['n_theta'], "nR": cfg['n_R'],
        "max_hd_points": cfg['max_hd_points']
    }

    train_dataset = DatasetGeoVAE3D(train_meta[0], train_meta[1], train_meta[2], shuffle=True, **ds_params)
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], num_workers=0)

    val_loader = None
    if len(val_meta[0]) > 0:
        val_dataset = DatasetGeoVAE3D(val_meta[0], val_meta[1], val_meta[2], shuffle=False, **ds_params)
        val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], num_workers=0)

    # 3. 实例化模型
    model_params = {
        "hidden_dim": cfg['model']['hidden_dim'],
        "latent_channels": cfg['model']['latent_channels'],
        "num_res_blocks": cfg['model']['num_res_blocks'],
        "compress_shape": tuple(cfg['model']['compress_shape']),
        "kl_weight": cfg['model']['kl_weight'],
        "hd_weight": cfg['model']['hd_weight'],
        "decay_rate": cfg['model']['decay_rate']
    }

    if cfg['field_name'] == "PORO":
        model = PoroVAE3D(poro_min=g_min, poro_max=g_max, **model_params)
    else:
        model = PermVAE3D(perm_min=g_min, perm_max=g_max, **model_params)

    # 4. Logger & 极致省空间的 Checkpoint
    tb_logger = TensorBoardLogger(save_dir=log_dir, name=f"{current_time}_{cfg['field_name'].lower()}_tb")
    csv_logger = CSVLogger(save_dir=log_dir, name=f"{current_time}_{cfg['field_name'].lower()}_csv")
    
    # 🌟 [修改] 自动覆盖，始终只保留 1 个最优模型，节省海量磁盘空间
    ckpt_filename = f"{current_time}_{cfg['field_name'].lower()}_best"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=ckpt_filename,
        save_top_k=1,                     # 永远只保留最好的那 1 个 epoch
        monitor="val_loss" if val_loader else "train_loss",
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

    logger.info("🔥 开始训练...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info("✅ 训练结束!")

if __name__ == "__main__":
    main()