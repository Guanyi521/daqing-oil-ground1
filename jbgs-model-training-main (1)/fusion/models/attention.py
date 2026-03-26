# 支持 FlashAttention、连续时间戳 RoPE 与 KV Cache 的特化 Attention 算子

import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeRoPE(nn.Module):
    """
    专为时空展平序列设计的旋转位置编码 (Time-specific RoPE)。
    完美支持基于真实生产日历的连续时间戳 (datetimes)，自动计算任意时间跨度的相对物理演化。
    """
    def __init__(self, head_dim: int, base: float = 100000.0):
        super().__init__()
        self.head_dim = head_dim
        # 计算逆频率 (Inverse Frequencies)，控制旋转角速率的衰减
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, datetimes: torch.Tensor, Ld: int):
        """
        前向传播计算时空旋转位置编码。
        
        Args:
            x (Tensor): 需要旋转的查询向量 Q 或键向量 K。形状: `(B, num_heads, nt * Ld, head_dim)`
            datetimes (Tensor): 当前处理时间步的真实时间戳/日龄数组，必须从小到大排列。形状: `(nt,)`, dtype: int64 或 float32
            Ld (int): 单个时间步的空间 Token 数量 (nz * n_theta * n_R)
            
        Returns:
            Tensor: 注入时间旋转角后的特征张量。形状: `(B, num_heads, nt * Ld, head_dim)`
        """
        device = x.device
        nt = datetimes.shape[0]
        
        # 1. 获取真实时间序列并转换为 float 参与三角函数计算
        t = datetimes.to(device).float()
        
        # 2. 计算时间步对应的旋转角度 (nt, head_dim / 2)
        freqs = torch.outer(t, self.inv_freq)
        
        # 3. LLaMA 风格：前后两半维度拼接 -> (nt, head_dim)
        freqs = torch.cat((freqs, freqs), dim=-1)
        
        # 4. 🌟 时空广播魔法：按空间维度 Ld 展开，保证同一时间切片内的所有空间 Token 共享相同的物理时间角
        # (nt, head_dim) -> (nt, 1, head_dim) -> (nt, Ld, head_dim) -> (nt * Ld, head_dim)
        freqs = freqs.unsqueeze(1).expand(-1, Ld, -1).reshape(nt * Ld, -1)
        
        # 5. 适配输入注意力矩阵的维度 -> (1, 1, nt * Ld, head_dim)
        freqs = freqs.unsqueeze(0).unsqueeze(0)
        
        cos = freqs.cos()
        sin = freqs.sin()

        # 6. 定义 LLaMA 风格的奇偶向量旋转函数
        def rotate_half(v):
            v1 = v[..., : v.shape[-1] // 2]
            v2 = v[..., v.shape[-1] // 2 :]
            return torch.cat((-v2, v1), dim=-1)

        # 7. 应用旋转矩阵
        x_rotated = (x * cos) + (rotate_half(x) * sin)
        return x_rotated.type_as(x)


class SelfAttention(nn.Module):
    """自注意力计算模块 (全解耦支持 FlashAttention, 连续时间 RoPE 与 KV Cache)"""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__() #
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除" #
        self.num_heads = num_heads #
        self.head_dim = d_model // num_heads #
        
        self.q_proj = nn.Linear(d_model, d_model) #
        self.k_proj = nn.Linear(d_model, d_model) #
        self.v_proj = nn.Linear(d_model, d_model) #
        self.out_proj = nn.Linear(d_model, d_model) #
        
        # 实例化基于真实日历的 TimeRoPE
        self.time_rope = TimeRoPE(self.head_dim)

    def forward(self, x: torch.Tensor, datetimes: torch.Tensor, mask: torch.Tensor = None, kv_cache: tuple = None):
        """
        前向传播计算自注意力。
        
        Args:
            x (Tensor): 输入主特征。形状: `(B, nt, Ld, d_model)`
            datetimes (Tensor): 对应 x 的绝对日龄/真实时间步序列。形状: `(nt,)`, dtype: int64
            mask (Tensor, optional): 自注意力掩码。形状: 允许传入未折叠的 `(nt, Ld, nt, Ld)`，或适应广播的形状。
            kv_cache (tuple, optional): 历史 KV 缓存。格式为 `(k_cache, v_cache)`
            
        Returns:
            tuple:
                - Tensor: 注意力输出。形状恢复为 `(B, nt, Ld, d_model)`
                - tuple: 更新后的 KV 缓存 `(k_new, v_new)`
        """
        B, nt, Ld, _ = x.shape
        L_tgt = nt * Ld
        
        # 1. 展平空间与时间维度 (B, nt*Ld, d_model)
        x_flat = x.view(B, L_tgt, -1)
        
        # 2. 投影并切分多头 -> 形状: (B, num_heads, nt*Ld, head_dim)
        q = self.q_proj(x_flat).view(B, L_tgt, self.num_heads, self.head_dim).transpose(1, 2) #
        k = self.k_proj(x_flat).view(B, L_tgt, self.num_heads, self.head_dim).transpose(1, 2) #
        v = self.v_proj(x_flat).view(B, L_tgt, self.num_heads, self.head_dim).transpose(1, 2) #

        # 3. 🌟 为当前处理的时间步的 Q 和 K 注入连续时间相对位置编码 (RoPE)
        q = self.time_rope(q, datetimes, Ld)
        k = self.time_rope(k, datetimes, Ld)

        # 4. KV Cache 拼接逻辑 (重点：缓存中的历史 K 早就被过去的时间戳旋转过了，直接拼接！)
        if kv_cache is not None: #
            k_cache, v_cache = kv_cache #
            k = torch.cat([k_cache, k], dim=2) #
            v = torch.cat([v_cache, v], dim=2) #
        
        new_kv_cache = (k, v) #

        # 5. Mask 形状展平自适应
        if mask is not None: #
            # 兼容处理用户未折叠的高维块状掩码: (nt, Ld, nt, Ld) -> (nt*Ld, nt*Ld)
            if mask.dim() == 4 and mask.shape == (nt, Ld, nt, Ld):
                mask = mask.view(L_tgt, L_tgt)
            # 适配 FlashAttention 的广播机制
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)

        # 6. PyTorch 原生 FlashAttention 极速计算
        attn_output = F.scaled_dot_product_attention( #
            q, k, v,  #
            attn_mask=mask,  #
            is_causal=False  # 因果掩码已融入外部结构性 mask，故保持 False
        )
        
        # 7. 重塑回原始的高维形状: (B, num_heads, nt*Ld, head_dim) -> (B, nt, Ld, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, nt, Ld, -1)
        return self.out_proj(attn_output), new_kv_cache #


class CrossAttention(nn.Module):
    """
    交叉注意力计算模块 (时间严格隔离版)
    利用“时间维折叠 (Time Folding)”技巧，完美实现对角块注意力 (Block-Diagonal Attention)，
    强制第 t 步的流体状态只关注第 t 步的外部特征，彻底杜绝物理因果泄漏。
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, feature: torch.Tensor, mask: torch.Tensor = None, kv_cache: tuple = None):
        """
        前向传播计算交叉注意力。
        
        Args:
            x (Tensor): 查询向量主特征。形状: `(B, nt, Ld, d_model)`
            feature (Tensor): 外部控制特征。🌟 必须统一形状为 `(B, nt, L_src, d_model)` 方便训练，预测时 nt 可取 1
                              (静态特征请在外层 expand 补齐 nt 维度)
            mask (Tensor, optional): 纯空间结构性掩码。形状: `(B, 1, Ld, L_src)` 或 `(1, 1, Ld, L_src)`
            kv_cache (tuple, optional): 推理时的缓存 (通常仅用于 nt=1 的推理加速)

        Returns:
            tuple:
                - Tensor: 注意力输出。形状恢复为 `(B, nt, Ld, d_model)`
                - tuple: 更新后的特征 KV 缓存
        """
        B, nt, Ld, _ = x.shape
        L_src = feature.shape[2]
        
        # ==========================================================
        # 🌟 核心魔法：时间维折叠 (Time Folding into Batch Dimension)
        # 将 (B, nt) 融合成伪批次维度，强制实现绝对的对角线块 (Block-Diagonal) 隔离！
        # ==========================================================
        B_pseudo = B * nt
        
        x_flat = x.view(B_pseudo, Ld, -1)
        q = self.q_proj(x_flat).view(B_pseudo, Ld, self.num_heads, self.head_dim).transpose(1, 2)

        # Cross-Attention 缓存逻辑
        # 注意：在自回归推理时，nt=1，此时 B_pseudo = B。
        # 对于动态特征，我们在推理的外层会主动清空它的 kv_cache，迫使它重新计算当步的 K, V。
        if kv_cache is not None:
            k, v = kv_cache
        else:
            feature_flat = feature.view(B_pseudo, L_src, -1)
            k = self.k_proj(feature_flat).view(B_pseudo, L_src, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(feature_flat).view(B_pseudo, L_src, self.num_heads, self.head_dim).transpose(1, 2)
            kv_cache = (k, v)

        # Mask 形状展自适应 (处理空间掩码)
        if mask is not None:
            # 假设外部传入的掩码是二维的纯空间拓扑掩码 (Ld, L_src)
            if mask.dim() == 2:
                # 适配 Flash Attention: (B_pseudo, num_heads, Ld, L_src)
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 4:
                # 如果传入的是 (B, 1, Ld, L_src)，需要扩展适应 B_pseudo
                # 这在每个 Batch 的拓扑结构不同时极其有用
                mask = mask.expand(-1, nt, -1, -1).reshape(B_pseudo, 1, Ld, L_src)

        # PyTorch 原生 FlashAttention 极速计算 (此时 L_tgt=Ld, L_source=L_src)
        # 因果隔离已经由 B_pseudo 物理层面保证，所以 is_causal=False 即可
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        
        # 重塑回多维时空结构 (B_pseudo, num_heads, Ld, head_dim) -> (B, nt, Ld, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, nt, Ld, -1)
        
        return self.out_proj(attn_output), kv_cache