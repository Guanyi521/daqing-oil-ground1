# layer struction for transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion.models.attention import SelfAttention, CrossAttention

# --------------------------------------------------------------------------------
#   layer struction for transformer
# --------------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    """标准解码器层 (包含自注意力机制与残差连接)"""
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.self_attention = SelfAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, datetimes: torch.Tensor, mask: torch.Tensor = None, kv_cache: tuple = None):
        """
        Args:
            x (Tensor): 输入特征。形状: `(B, nt, Ld, d_model)`
            datetimes (Tensor): 对应 x 的绝对日龄/真实时间步序列。形状: `(nt,)`, dtype: int64
            mask (Tensor, optional): 自注意力掩码：因果掩码 & 结构性掩码。形状: `(nt, Ld, nt, Ld)`
            kv_cache (tuple, optional): 历史 KV 缓存。

        Returns:
            tuple: 输出特征张量与更新后的 KV 缓存。
        """
        attn_out, new_kv = self.self_attention(self.norm(x), datetimes, mask=mask, kv_cache=kv_cache)
        return x + attn_out, new_kv


class MultiFeatureFusion(nn.Module):
    """多特征解耦融合层 (级联残差特征注入)"""
    def __init__(self, d_model: int, head_splits: list):
        super().__init__()
        self.d_model = d_model
        self.n_features = len(head_splits)
        # 1. 独立的交叉注意力模块
        self.cross_attentions = nn.ModuleList([
            CrossAttention(d_model, num_heads) for num_heads in head_splits
        ])

        # 2. 独立的主干 Query 归一化层 (Pre-Norm)
        self.q_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_features)])

        # 3. 🌟 独立的外部特征 K/V 归一化层
        # 完美解决你担心的“多种融合特征尺度不一致”问题！
        self.kv_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.n_features)])

    def forward(self, x: torch.Tensor, features: torch.Tensor, mask: torch.Tensor = None, cross_caches: list = None):
        """
        建议：把相对静态、基础的特征（如地质）放在循环的前面；把高频、动态、起决定性冲击作用的特征（如井控）放在循环的最后面。这样网络能基于最稳固的地质底座，去响应井控的脉冲。

        Args:
            x (Tensor): 主线自回归状态。形状: `(B, nt, Ld, d_model)`
            features (Tensor): 多模态特征集合 (如 地质, 井控)。形状: `(B, n_features, L_src, d_model)`
            mask (Tensor, optional): 交叉注意力掩码：结构性掩码。形状: `(nt, Ld, L_src)`
            cross_caches (list, optional): 各个特征通道的静态 KV 缓存列表。

        Returns:
            tuple:
                - Tensor: 级联融合后的特征。形状: `(B, nt, Ld, d_model)`
                - list: 更新后的特征 KV 缓存列表。
        """
        if cross_caches is None:
            cross_caches = [None] * self.n_features

        new_cross_caches = []

        # 级联迭代 (Cascade Iteration)
        for i in range(self.n_features):
            # 提取第 i 个外部特征
            # feature_i = features[:, i, :, :]
            feature_i = features[:, i, :, :, :]
            # 🌟 对外部特征进行独立的 LayerNorm，将其强行拉回 N(0, 1) 的标准尺度
            # 无论地质场和井控场的原始数值差异多大，到这里众生平等！
            normed_feature_i = self.kv_norms[i](feature_i)

            # 主动态变化同样要正则
            normed_x_i = self.q_norms[i](x)
            
            # 获取缓存
            c_cache = cross_caches[i]
            
            # 提取信息：Query 使用 q_norms，Key/Value 使用刚刚的 normed_feature_i
            attn_out, new_c_cache = self.cross_attentions[i](
                normed_x_i, 
                normed_feature_i, 
                mask=mask, 
                kv_cache=c_cache
            )
            
            # 🚨 修复致命 Bug：立刻把提炼出的信息加到主干 x 上！
            # 这样在下一次循环 (i+1) 时，网络就是带着上一层融合好的认知去查新特征了。
            x = x + attn_out
            
            new_cross_caches.append(new_c_cache)

        # 返回的是已经像滚雪球一样积累了所有模态信息的最终 x
        return x, new_cross_caches
    
    
class FNN(nn.Module):
    """前馈神经网络 (FFN)"""
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(), # GELU 通常比 ReLU 在大模型中表现更好
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x: torch.Tensor):
        return x + self.net(x)