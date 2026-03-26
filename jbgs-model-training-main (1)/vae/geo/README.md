## 🛢️ Geo VAE 3D - 地质静态场空间表征模块

### 🗺️ 子模块总览 (Methodology Overview)

本模块核心方法论如下图所示：

![Overview of RFM Methodology](./assets/geo_vae.png)


本模块用于将 Eclipse 模拟器输出的 3D 地质静态场（孔隙度 PORO、渗透率 PERMX）映射为基于物理柱状坐标的 3D 潜变量空间 (Latent Space Grid)。提取后的特征将用于后续流体动力学大模型的 Cross-Attention 条件注入。

---

### 📂 子目录结构 (Sub-Project Structure)

```Catalog
vae/
├── layers.py                        # 跨目录底层算子 (3D柱坐标卷积、环形物理插值、缩放器)
└── geo/
    ├── dataset_geo_vae.py           # 基于 IterableDataset 的分块热加载引擎 (支持井级拆分)
    ├── geo_vae.py                   # 物理感知 VAE 核心 (含 PoroVAE3D / PermVAE3D)
    ├── pt_train_stage1_geo_vae.py   # PyTorch Lightning 工业级训练入口
    ├── vae_geo_perm_config.yaml     # 渗透率 (PERMX) 训练参数配置
    └── vae_geo_poro_config.yaml     # 孔隙度 (PORO) 训练参数配置
```

---

### 🛠️ 模块说明

-`geo_vae.py`: 实现了极严格的物理边界约束。采用 `LogScaler` 处理渗透率的对数量级差异；采用 `Masked` 机制免疫死网格干扰；利用高级索引实现了无显存广播的硬数据保真损失 (`hd_loss`)。

`dataset_geo_vae.py`: 由于油藏项目规模庞大，数据集采用 `nums_loading_projects` 参数控制内存持水量。每次只在内存中解压指定数量的项目，训练完毕后自动回收。

`pt_train_stage1_geo_vae.py`: 在初始化时自动全局扫描数据的 `Min/Max` 极值，并按照 `train_val_split` 比例，**在同一油田内部按“井”进行物理切分**，确保模型学到的是插值泛化能力，而非死记硬背。

---

### 🚀 启动训练

* 为确保跨平行文件夹导包正确，请在 `vae` 的**上一级目录**（或项目根目录）利用 `-m` 参数启动脚本。

- **训练渗透率模型**
```Bash
python -m vae.geo.pt_train_stage1_geo_vae --config vae/geo/vae_geo_perm_config.yaml
```
- **训练孔隙度模型**
```Bash
python -m vae.geo.pt_train_stage1_geo_vae --config vae/geo/vae_geo_poro_config.yaml
```

**日志与监控 (Logging):**

所有文件自动存放在 `vae/geo/log/` 与 `vae/geo/checkpoints/` 下。

控制台同步生成 `.log` 文本日志。

训练中途，新开终端运行 `tensorboard --logdir=vae/geo/log` 可实时查看 Loss 曲线。

训练节约磁盘空间 **Checkpoint** 将自动覆盖保存为 `[时间戳]_[静态场名]_best.ckpt`。

---

### 📈 超参调试与模型容量评估策略 (Best Practices)

地质场非均质性极强，盲目推高 Epoch 会导致算力浪费。建议通过观察全局扫描的日志，采取 **"先宽后深"** 的策略调试。

- **阶段 1：数据规模侦测与短跑试水 (Epochs: 10~20)**

启动脚本时，观察控制台输出的物理规模估计：

|`📊 扫描完毕! 总有效网格: 2,500,000 | 井数: 150`

如果网格总量 > 500万 且 井数 > 500，默认的轻量级参数必然会欠拟合。先用当前 YAML 跑 20 个 Epoch，如果在 TensorBoard 中发现 `val_recon_loss` 早早走平，且生成的特征极度模糊，进入阶段 2。

- **阶段 2：网络容量与边界博弈 (Epochs: 50~100)**

在 `vae_geo_XXX_config.yaml` 的 `model` 节点下调节：

1. 加宽信息通道 (`latent_channels`)：这是解决模糊最有效的手段。若渗透率重建细节丢失严重，将 latent_channels 从 8 提至 16。

2. 加深非线性推理 (`num_res_blocks`)：若发现网络无法拟合大数值区域，将残差块深度从 1 提升至 2 或 3，增加网络的表达容量。

3. 缓解后验坍塌 (`kl_weight`)：如果网络过度关注 **$N(0,1)$** 的分布导致 `val_recon_loss` 降不下去，尝试将 `kl_weight` 从 `1.0e-5` 下调至 `1.0e-6`，向重建妥协。

4. 井点模糊 (`hd_weight`)：若大面积趋势正确，但单井所处网格的渗透率与硬数据匹配不佳，将 `hd_weight` 适当调高（如 20.0）。

*操作：选定参数组合，运行 50 个 Epoch，对比不同参数版本（`_tb` 文件夹中自动记录版本号）的 `val_recon_loss` 下降斜率，锁定最佳组合。*

- **阶段 3：全量榨取 (Epochs: 300~500+)**

确定最优超参后，将 `epochs` 拉满。如果内存充裕，可适度调大 `nums_loading_projects`，增加每个 Batch 中的异质性，以提升最终模型的泛化上限。

---
*Training for geologic VAE of Next-Generation Reservoir Surrogate.*