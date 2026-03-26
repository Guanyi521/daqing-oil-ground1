# 主体网络：DecoderFusionTransformer

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math  # 必须引入 math
from fusion.models.layers import  DecoderLayer, MultiFeatureFusion, FNN
from fusion.models.embeddings import (
    generate_euclidean_positional_encoding,
    generate_block_causal_mask
)


# Transformer Class that handles multi-encoder feature fusion
class DecoderFusionTransformer(pl.LightningModule):
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        num_heads: int, 
        head_splits: list, 
        num_layers: int, 
        hidden_dim: int,
        max_nt_width: int, # 新增参数：训练时的最大时间窗口
        Ld: int,           # 新增参数：单步空间长度
        grid_shape: tuple, # 柱状坐标形状 grid_shape (tuple): 例如 (nz/2, n_theta/2, n_R/2)
    ):
        super().__init__()
        self.save_hyperparameters()

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 添加一个可学习的缩放系数，解决 x 与 pos_enc 的尺度冲刷问题！
        self.pos_scale = nn.Parameter(torch.ones(1))      # 空间缩放
        self.time_scale = nn.Parameter(torch.ones(1))     # 时间缩放
        self.emb_dropout = nn.Dropout(0.1)

        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads) for _ in range(num_layers)])
        self.fusion_layers = nn.ModuleList([MultiFeatureFusion(d_model, head_splits) for _ in range(num_layers)])
        self.ffn_layers = nn.ModuleList([FNN(d_model, hidden_dim) for _ in range(num_layers)])
        
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.grid_shape = grid_shape

        # 生成基于最大时间窗口的完整 Block Causal Mask
        full_mask = generate_block_causal_mask(max_nt_width, Ld, 'cpu')

        # 注册为 Buffer (不会被 Optimizer 更新，但会随着 model.to('cuda') 自动转移到 GPU)
        self.register_buffer("full_mask", full_mask)

    def forward(
        self, 
        x_indices: torch.Tensor, 
        datetimes: int,
        features: torch.Tensor, 
        maxR: float,          # 物理超参数: 例如 1000.0 (米) 用于欧式空间位置编码
        total_z_depth: float, # 物理超参数: 例如 50.0 (米) 用于欧式空间位置编码
        self_mask: torch.Tensor = None, 
        cross_mask: torch.Tensor = None, 
        past_kv_caches: list = None, 
        past_cross_caches: list = None,
    ) -> tuple:
        """
        Transformer 主前向传播函数 (解耦维度 + 完整 KV Cache 版)。
        
        Args:
            x_indices (Tensor[int64]): 形状 `(B, nt, Ld)`
            datetimes (Tensor[int64] | Tensor[float32]): 形状 `(nt,)` 当前处理时间步的真实时间戳/日龄数组，必须从小到大排列
            features (Tensor[float32]): 形状 `(B, n_features, L_src, d_model)`
            maxR (float): 物理超参数: 例如 1000.0 (米) 用于欧式空间位置编码
            total_z_depth (float): 物理超参数: 例如 50.0 (米) 用于欧式空间位置编码
            self_mask (Tensor[bool], optional): 自注意力掩码: 因果掩码 & 结构性掩码; 形状 `(nt * Ld, nt * Ld)=(L_tgt, L_tgt)`
            cross_mask (Tensor, optional): 交叉注意力掩码: 结构性掩码; 形状: `(nt * Ld, L_src)=(L_tgt, L_src)`
            past_kv_caches (list, optional): 自注意力历史缓存
            past_cross_caches (list, optional): 交叉注意力静态缓存


        Returns:
            logits (Tensor[float32]): 形状 `(B, nt, Ld, vocab_size)`
            new_kv_caches (list)
            new_cross_caches (list)
        """
        B, nt, Ld = x_indices.shape
        
        # 词嵌入并放大数值 (防止被位置编码冲刷)
        # 形状: (B, nt, Ld) -> (B, nt, Ld, d_model)
        x = self.token_emb(x_indices) * math.sqrt(self.hparams.d_model)

        # 柱状坐标->转->欧式坐标位置编码 (形状: (Ld, d_model))
        # spat_pe = generate_euclidean_positional_encoding(self.grid_shape, self.hparams.d_model, x.device, maxR, total_z_depth)
        spat_pe = generate_euclidean_positional_encoding(
            self.grid_shape,
            self.hparams.d_model,
            maxR,
            total_z_depth,
            x.device
        )
        # 维度广播极其优雅的加法
        # 巧妙利用 unsqueeze(1) 将空间 PE 变成 (1, 1, Ld, d_model)
        # 这样它可以直接和 (B, nt, Ld, d_model) 的 x 相加，PyTorch 会在 nt 维度上自动广播！
        x = x + self.pos_scale * spat_pe.unsqueeze(0).unsqueeze(0)
        # x = self.emb_dropout(x).view(B, nt * Ld, self.hparams.d_model)
        x = self.emb_dropout(x)
        # 🌟 为外部 Features 注入纯 [空间] 编码
        # features 形状: (B, n_features, L_src, d_model)
        # spatial_pos_enc 形状: (1, Ld, d_model) -> 扩展为 (1, 1, L_src, d_model)
        # 前提是 L_src == Ld，如果拓扑一致，这是绝对正确的！
        # features = features + self.pos_scale * self.spatial_pos_enc.unsqueeze(1)
        if features.shape[2] != Ld:
            raise ValueError(f"L_src ({features.shape[2]}) must equal Ld ({Ld}) when sharing spatial PE.")

        features = features + self.pos_scale * spat_pe.unsqueeze(0).unsqueeze(0)
        features = features.unsqueeze(2).expand(-1, -1, nt, -1, -1)
        # 展平时间与空间维度，喂给 Transformer 骨干
        # 形状: (B, nt, Ld, d_model) -> (B, nt * Ld, d_model)
        # x = x.view(B, nt * Ld, self.hparams.d_model)

        # 初始化用于存放当前步缓存的空列表
        new_kv_caches = []
        new_cross_caches = []
        if past_kv_caches is None:
            past_kv_caches = [None] * len(self.decoder_layers)
        if past_cross_caches is None:
            past_cross_caches = [None] * len(self.fusion_layers)

        # 如果外部没有传 mask（比如是在推理预测时，长度为 1），就不要用块状掩码
        # 如果是训练阶段，直接切出当前长度 (nt * Ld) 的局部掩码
        
        # 融合因果掩码和传进来的空间自我掩码
        final_self_mask = None
        if nt > 1:
            seq_len = nt * Ld
            # 取出因果掩码 (True=遮蔽)
            causal_mask = self.full_mask[:seq_len, :seq_len]
            
            # SDPA 接受的布尔 mask 必须是: True=允许注意, False=遮蔽
            # 所以我们要取 causal_mask 的反 (~causal_mask)
            causal_allowed = ~causal_mask

            # 🌟 修复：Self_mask 传入的是 (B, 1, 1, seq_len)，causal_allowed 是 (seq_len, seq_len)
            # 广播后 final_self_mask 形状为 (B, 1, seq_len, seq_len)，完美适配 SDPA 多头！
            final_self_mask = causal_allowed.unsqueeze(0).unsqueeze(0)
            
            if self_mask is not None:
                # 只有符合因果律 (causal_allowed) 且符合空间有效性 (self_mask) 的位置才允许注意
                final_self_mask = final_self_mask & self_mask
           

        # Transformer 骨干网络 (级联自回归 + 特征注入)
        for i, (self_decode, cross_fusion, ffn) in enumerate(zip(self.decoder_layers, self.fusion_layers, self.ffn_layers)):
            
            x, new_kv = self_decode(x, datetimes, mask=final_self_mask, kv_cache=past_kv_caches[i])
            new_kv_caches.append(new_kv)

            x, new_cross = cross_fusion(x, features, mask=cross_mask, cross_caches=past_cross_caches[i])
            new_cross_caches.append(new_cross)

            x = ffn(x)

        # 映射到 VQ Codebook 词表概率分布
        # 形状: (B, nt * Ld, d_model) -> (B, nt * Ld, vocab_size)
        logits = self.lm_head(x)
        
        # 优雅地 Reshape 回解耦维度！
        # 形状: (B, nt, Ld, vocab_size)
        logits = logits.view(B, nt, Ld, -1)

        return logits, new_kv_caches, new_cross_caches
    
    def _training_step(self, window_tokens, datetimes, features, spatial_mask, feature_mask, maxR, total_z_depth):

        """
        支持多时间步滑动窗口 (nt_width) 的极速并行训练。
        Args:
            window_tokens: int64(B, nt_width+1, Ld)
            datetimes: float32(B, nt_width)
            features float32(B, n_features, L_src, d_model)
            spatial_mask: bool(B, Ld) 有效区域为 True，无效区域 False
            feature_mask: bool(B, L_src) 有效区域为 True，无效区域 False
            maxR (float): 物理超参数: 例如 1000.0 (米) 用于欧式空间位置编码
            total_z_depth (float): 物理超参数: 例如 50.0 (米) 用于欧式空间位置编码
        Returns:
            loss: float
            logits (Tensor[float32]): 形状 `(B, nt, Ld, vocab_size)`
        """

        x_indices = window_tokens[:, :-1]

        target_indices = window_tokens[:, 1:]

        B, nt, Ld = x_indices.shape

        sp_mask_flat = spatial_mask.unsqueeze(1).expand(B, nt, Ld).reshape(B, nt * Ld)

        # 🌟 绝杀：只屏蔽 Key，且补充两个 1 供 SDPA 的 heads 和 queries 进行广播！
        # 形状: (B, 1, 1, nt*Ld)
        self_mask = sp_mask_flat.unsqueeze(1).unsqueeze(1)

        # 形状: (B, 1, 1, L_src)
        # cross_mask = feature_mask.unsqueeze(1).unsqueeze(1)
        cross_mask = feature_mask[:, None, None, :].expand(-1, nt, Ld, -1)
        # 前向传播
        # logits, _, _ = self(x_indices, datetimes, features, self_mask, cross_mask, maxR, total_z_depth)
        logits, _, _ = self(
            x_indices=x_indices,
            datetimes=datetimes,
            features=features,
            maxR=maxR,
            total_z_depth=total_z_depth,
            self_mask=self_mask,
            cross_mask=cross_mask
        )
        # ==========================================
        # 核心算子形状对齐 (F.cross_entropy)
        # logits 展平为 (B * nt * Ld, vocab_size)
        # targets 展平为 (B * nt * Ld)
        # ==========================================
        # 直接计算 Loss，PyTorch 会自动忽略所有 0 的位置
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), # logits 没做切片，本来就是连续的，view 极速零拷贝
            target_indices.reshape(-1),       # logits 没做切片，本来就是连续的，view 极速零拷贝
            ignore_index=0  # 显式声明（默认也是0）
        )

        return loss, logits


    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        支持多时间步滑动窗口 (nt_width) 的极速并行训练。
        Args:
            batch数据类型约定: 
            -window_tokens(int64),
            -features(float32), 
            -grid_shape(int,int,int), 
            -spatial_mask(bool), 
            -feature_mask(bool),
            -maxR(float),
            -total_z_depth(float)
        Returns:
            loss: float
        """

        window_tokens, datetimes, features, spatial_mask, feature_mask, maxR, total_z_depth = batch

        loss, _ = self._training_step(window_tokens, datetimes,features, spatial_mask, feature_mask, maxR, total_z_depth)

        self.log('train_loss', loss, prog_bar=True)
        return loss
    

if __name__ == "__main__":
    import torch
    
    print("="*60)
    print("🚀 启动 DecoderFusionTransformer 本地极限测试...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前运行设备: {device}")
    
    # ---------------- 测试参数设定 ----------------
    B = 2                   # Batch size
    nt_width = 4            # 滑动窗口步数 (传进去会是 nt_width + 1 = 5)
    nz, nθ, nR = 10, 10, 5 
    Ld = nz * nθ * nR # 单步空间 Token 数量 = 500
    
    d_model = 128
    
    # 【核心还原】 3 个特征，头数分别占用 4, 2, 8
    # 注意: d_model (128) 必须能被列表里每一个元素整除！
    head_splits = [4, 2, 8] 
    n_features = len(head_splits) # 也就是 3
    print(f"🧬 特征融合通道已开启: {n_features} 组特征，独立头数分别为 {head_splits}")

    vocab_size = 1025
    L_src = 500              # 地质/井控/边界等特征的空间长度
    maxR, total_z_depth = 1000.0, 50.0 # 物理超参数: 例如 1000.0 (米) 用于欧式空间位置编码
    
     
    # 1. 实例化主干模型
    # 确保初始化时传入正确的 head_splits
    model = DecoderFusionTransformer(
        vocab_size=vocab_size, 
        d_model=d_model, 
        num_heads=4,             # 这个 num_heads 专用于 SelfAttention
        head_splits=head_splits, # 专用于 CrossAttention 的并行解耦
        num_layers=2, 
        hidden_dim=256, 
        max_nt_width=10, 
        Ld=Ld,
        grid_shape = (nz, nθ, nR),
    ).to(device)
    
    # 2. 构造 100% 保留 n_features 维度的虚拟数据
    window_tokens = torch.randint(0, vocab_size, (B, nt_width + 1, Ld), device=device)
    
    datetimes = torch.arange(nt_width, device=device).float() # 模拟连续时间步的真实时间戳/日龄，形状 (nt_width,)

    # 🌟 绝不省维度：(B, n_features, L_src, d_model) -> (2, 3, 50, 128)
    features = torch.randn(B, n_features, L_src, d_model, device=device) 
    
    
    # 3. 构造掩码
    spatial_mask = (torch.rand(B, Ld, device=device) > 0.2)
    feature_mask = (torch.rand(B, L_src, device=device) > 0.1) # 假定 3 个特征共享同一个结构性有效掩码
    
    batch = (window_tokens, datetimes, features, spatial_mask, feature_mask, maxR, total_z_depth)

    # 4. 执行前向传播
    print("\n⏳ 开始前向传播与多特征 Attention 计算...")
    model.log = lambda *args, **kwargs: None
    loss = model.training_step(batch, batch_idx=0)
    print("✅ 前向传播测试 100% 通过！矩阵与多特征广播全线对齐！")
    print(f"🎯 最终 CrossEntropy Loss: {loss.item():.4f}")
        
    # 5. 显存监控 (通常会在 1.5 GB 左右，绝对安全)
    if torch.cuda.is_available():
        print(f"📊 峰值显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print("="*60)