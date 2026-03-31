import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from vae.vq.vq_vae import VQVAE3D
from fusion.models.transformer import DecoderFusionTransformer


class LatentDynamicsPipeline(pl.LightningModule):
    """
    统一动力学流水线 (Unified Dynamics Pipeline)

    支持三种静态特征来源：
    1. 在线 GeoVAE 编码:
       poro_field / perm_field -> encode_latent -> static_features
    2. 预提取 latent:
       z_poro / z_perm / latent_mask -> static_features
    3. 旧版直接传 features

    动态特征:
    - 若不给，则自动补零占位
    """

    def __init__(
        self,
        vqvae_model: nn.Module,
        transformer_model: nn.Module,
        poro_vae: nn.Module = None,
        perm_vae: nn.Module = None,
        d_model: int = 128,
        freeze_vqvae: bool = True,
        freeze_static_vae: bool = True,
        dummy_dynamic_groups: int = 1,
    ):
        super().__init__()
        self.vqvae = vqvae_model
        self.transformer = transformer_model

        self.poro_vae = poro_vae
        self.perm_vae = perm_vae
        self.d_model = d_model
        self.freeze_vqvae = freeze_vqvae
        self.freeze_static_vae = freeze_static_vae
        self.dummy_dynamic_groups = dummy_dynamic_groups

        if self.freeze_vqvae:
            self._freeze_module(self.vqvae)

        if self.freeze_static_vae:
            if self.poro_vae is not None:
                self._freeze_module(self.poro_vae)
            if self.perm_vae is not None:
                self._freeze_module(self.perm_vae)

        self.use_geo_static = (self.poro_vae is not None) and (self.perm_vae is not None)
        if self.use_geo_static:
            poro_latent_channels = int(self.poro_vae.hparams.latent_channels)
            perm_latent_channels = int(self.perm_vae.hparams.latent_channels)

            self.poro_proj = nn.Linear(poro_latent_channels, d_model)
            self.perm_proj = nn.Linear(perm_latent_channels, d_model)
        # 🌟 新增：动态井控特征的投影层
        # 假设 Dataset 传来的原始控制特征维度是 12
        self.dynamic_feature_dim = 12
        self.dynamic_proj = nn.Linear(self.dynamic_feature_dim, d_model)

    def _freeze_module(self, module: nn.Module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vqvae:
            self.vqvae.eval()
        if self.freeze_static_vae:
            if self.poro_vae is not None:
                self.poro_vae.eval()
            if self.perm_vae is not None:
                self.perm_vae.eval()
        return self

    # =========================================================
    # 1. 在线 GeoVAE latent -> static_features
    # =========================================================
    @torch.no_grad()
    def encode_static_features(self, poro_field, perm_field, spatial_mask):
        """
        使用 poro_vae / perm_vae 提取 latent，并构造成:
            static_features: (B, 2, L_src, d_model)
            feature_mask:    (B, L_src)

        Args:
            poro_field:   (B, 1, nz, n_theta, n_R)
            perm_field:   (B, 1, nz, n_theta, n_R)
            spatial_mask: (B, nz, n_theta, n_R) 或 (B, 1, nz, n_theta, n_R)
        """
        if not self.use_geo_static:
            raise RuntimeError("当前 pipeline 未传入 poro_vae / perm_vae，无法在线生成静态特征。")

        if spatial_mask.dim() == 4:
            spatial_mask_5d = spatial_mask.unsqueeze(1).float()
        elif spatial_mask.dim() == 5:
            spatial_mask_5d = spatial_mask.float()
        else:
            raise ValueError(f"spatial_mask 维度不正确: {tuple(spatial_mask.shape)}")

        z_poro, latent_mask_poro = self.poro_vae.encode_latent(
            poro_field, spatial_mask_5d, use_mean=True
        )
        z_perm, latent_mask_perm = self.perm_vae.encode_latent(
            perm_field, spatial_mask_5d, use_mean=True
        )

        return self.encode_static_features_from_latent(
            z_poro=z_poro,
            z_perm=z_perm,
            latent_mask=latent_mask_poro
        )

    # =========================================================
    # 2. 预提取 latent -> static_features
    # =========================================================
    @torch.no_grad()
    def encode_static_features_from_latent(self, z_poro, z_perm, latent_mask):
        """
        直接使用预提取 latent 构造 static_features

        Args:
            z_poro:      (B, C_poro, nz_lat, n_theta_lat, n_R_lat)
            z_perm:      (B, C_perm, nz_lat, n_theta_lat, n_R_lat)
            latent_mask: (B, 1, nz_lat, n_theta_lat, n_R_lat)

        Returns:
            static_features: (B, 2, L_src, d_model)
            feature_mask:    (B, L_src)
        """
        if not self.use_geo_static:
            raise RuntimeError("当前 pipeline 未传入 poro_vae / perm_vae，无法构建静态特征投影层。")

        if z_poro.shape[2:] != z_perm.shape[2:]:
            raise ValueError(
                f"z_poro 空间尺寸 {tuple(z_poro.shape[2:])} 与 z_perm 空间尺寸 {tuple(z_perm.shape[2:])} 不一致"
            )

        if latent_mask.dim() != 5:
            raise ValueError(f"latent_mask 维度应为 5，收到 {tuple(latent_mask.shape)}")

        B = z_poro.shape[0]
        C_poro = z_poro.shape[1]
        C_perm = z_perm.shape[1]

        # (B, C, nz_lat, n_theta_lat, n_R_lat) -> (B, L_src, C)
        z_poro = z_poro.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C_poro)
        z_perm = z_perm.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, C_perm)

        f_poro = self.poro_proj(z_poro)   # (B, L_src, d_model)
        f_perm = self.perm_proj(z_perm)   # (B, L_src, d_model)

        static_features = torch.stack([f_poro, f_perm], dim=1)  # (B, 2, L_src, d_model)
        feature_mask = latent_mask.flatten(1).bool()            # (B, L_src)

        return static_features, feature_mask

    def build_dummy_dynamic_features(self, B, nt_init, L_src, d_model, device, dtype):
        return torch.zeros(
            B,
            nt_init,
            self.dummy_dynamic_groups,
            L_src,
            d_model,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def build_features_from_geo(self, poro_field, perm_field, spatial_mask, nt_init, dynamic_features=None):
        static_features, feature_mask = self.encode_static_features(
            poro_field=poro_field,
            perm_field=perm_field,
            spatial_mask=spatial_mask
        )

        B, _, L_src, d_model = static_features.shape

        # ====== 替换开始：动态特征处理逻辑 ======
        if dynamic_features is None:
            # 兼容旧逻辑：如果真的传了 None，就用全零的 Dummy 填充
            dynamic_features = self.build_dummy_dynamic_features(
                B=B, nt_init=nt_init, L_src=L_src, d_model=d_model,
                device=static_features.device, dtype=static_features.dtype
            )
        elif dynamic_features.dim() == 3:
            # 🌟 新增核心逻辑：接收 Dataset 传来的 (Batch, T, 12) 真实井控数据
            B_dyn, T_dyn, dim_dyn = dynamic_features.shape

            # 1. 线性升维：把 12 维投影到 d_model (例如 128 维)
            # 形状变为: (B, T, d_model)
            dyn_emb = self.dynamic_proj(dynamic_features.to(static_features.dtype).to(static_features.device))

            # 2. 增加空间维度：为了和静态特征拼接，增加 groups 和 L_src 维度
            # 形状变为: (B, T, 1, 1, d_model)
            dyn_emb = dyn_emb.view(B_dyn, T_dyn, 1, 1, d_model)

            # 3. 空间广播：把这个指令广播到所有的 L_src 空间网格上
            # 形状变为: (B, T, 1, L_src, d_model)
            dynamic_features = dyn_emb.expand(B_dyn, T_dyn, 1, L_src, d_model)

        # 兜底拦截：如果经过上述处理还不是 5D，就报错
        if dynamic_features.dim() != 5:
            raise ValueError(
                f"build_features_from_geo 期望 dynamic_features 维度为 5，收到 {tuple(dynamic_features.shape)}"
            )
        # ====== 替换结束 ======

        # 截取最后一个时间步的动态特征（原作者逻辑保持不变）
        dynamic_last = dynamic_features[:, -1, ...]  # (B, 1, L_src, d_model)

        # 将静态地质底座和动态井控指令拼接！
        features = torch.cat([static_features, dynamic_last], dim=1)

        return features, static_features, dynamic_features, feature_mask

    @torch.no_grad()
    def build_features_from_latent(self, z_poro, z_perm, latent_mask, nt_init, dynamic_features=None):
        static_features, feature_mask = self.encode_static_features_from_latent(
            z_poro=z_poro,
            z_perm=z_perm,
            latent_mask=latent_mask
        )

        B, _, L_src, d_model = static_features.shape

        # ====== 🌟 把 geo 里的那段逻辑原样复制过来 ======
        if dynamic_features is None:
            dynamic_features = self.build_dummy_dynamic_features(
                B=B, nt_init=nt_init, L_src=L_src, d_model=d_model,
                device=static_features.device, dtype=static_features.dtype
            )
        elif dynamic_features.dim() == 3:
            B_dyn, T_dyn, dim_dyn = dynamic_features.shape
            dyn_emb = self.dynamic_proj(dynamic_features.to(static_features.dtype).to(static_features.device))
            dyn_emb = dyn_emb.view(B_dyn, T_dyn, 1, 1, d_model)
            dynamic_features = dyn_emb.expand(B_dyn, T_dyn, 1, L_src, d_model)

        if dynamic_features.dim() != 5:
            raise ValueError(
                f"build_features_from_latent 期望 dynamic_features 维度为 5，收到 {tuple(dynamic_features.shape)}"
            )
        # =================================================

        dynamic_last = dynamic_features[:, -1, ...]  # (B, n_dynamic, L_src, d_model)
        features = torch.cat([static_features, dynamic_last], dim=1)

        return features, static_features, dynamic_features, feature_mask

    # =========================================================
    # 3. Forward
    # =========================================================
    @torch.no_grad()
    def forward(
        self,
        x_init_field,
        datetimes,
        static_features,
        dynamic_features,
        spatial_mask,
        feature_mask,
        maxR,
        total_z_depth,
        steps=1
    ):
        self.eval()
        B, nt_init, C, nz, n_theta, n_R = x_init_field.shape

        x_folded = x_init_field.reshape(B * nt_init, C, nz, n_theta, n_R)
        mask_folded = spatial_mask.unsqueeze(1).unsqueeze(1).expand(
            B, nt_init, 1, nz, n_theta, n_R
        ).reshape(B * nt_init, 1, nz, n_theta, n_R)

        init_indices, _, _ = self.vqvae.encode_to_indices(x_folded, mask_folded)

        nz_lat, n_theta_lat, n_R_lat = init_indices.shape[1:]
        Ld = nz_lat * n_theta_lat * n_R_lat

        curr_tokens = init_indices.view(B, nt_init, Ld)

        latent_mask = (curr_tokens[:, 0, :] != 0).bool()
        feature_mask = feature_mask.bool()

        if static_features.shape[2] != Ld:
            raise ValueError(
                f"静态特征 L_src={static_features.shape[2]} 与动态 token Ld={Ld} 不一致。"
            )

        past_kv_caches = None
        past_cross_caches = None
        generated_tokens = []

        max_nt_width = self.transformer.hparams.max_nt_width

        for step_idx in range(steps):
            sp_mask_flat = latent_mask.unsqueeze(1).expand(B, curr_tokens.shape[1], Ld).reshape(B, -1)
            self_mask = sp_mask_flat.unsqueeze(1).unsqueeze(1)
            cross_mask = feature_mask[:, None, None, :].expand(-1, curr_tokens.shape[1], Ld, -1)

            if dynamic_features.dim() == 5:
                use_t = min(curr_tokens.shape[1] - 1, dynamic_features.shape[1] - 1)
                dynamic_step_features = dynamic_features[:, use_t, ...]
            elif dynamic_features.dim() == 4:
                dynamic_step_features = dynamic_features
            else:
                raise ValueError(f"dynamic_features 维度不正确: {tuple(dynamic_features.shape)}")

            if dynamic_step_features.shape[2] != Ld:
                raise ValueError(
                    f"动态特征 L_src={dynamic_step_features.shape[2]} 与动态 token Ld={Ld} 不一致。"
                )

            features = torch.cat([static_features, dynamic_step_features], dim=1)

            logits, past_kv_caches, past_cross_caches = self.transformer(
                x_indices=curr_tokens,
                datetimes=datetimes,
                features=features,
                maxR=maxR,
                total_z_depth=total_z_depth,
                self_mask=self_mask,
                cross_mask=cross_mask,
                past_kv_caches=past_kv_caches,
                past_cross_caches=past_cross_caches,
            )

            next_logits = logits[:, -1, :, :]
            next_tokens = torch.argmax(next_logits, dim=-1)
            next_tokens = next_tokens.masked_fill(~latent_mask, 0)

            generated_tokens.append(next_tokens.unsqueeze(1))

            if past_kv_caches is not None and past_kv_caches[0][0].shape[1] >= max_nt_width:
                past_kv_caches = [
                    (k[:, -max_nt_width + 1:, ...], v[:, -max_nt_width + 1:, ...])
                    for (k, v) in past_kv_caches
                ]
                curr_tokens = next_tokens.unsqueeze(1)
            else:
                curr_tokens = torch.cat([curr_tokens, next_tokens.unsqueeze(1)], dim=1)

        gen_tokens_tensor = torch.cat(generated_tokens, dim=1)
        gen_indices_folded = gen_tokens_tensor.view(B * steps, nz_lat, n_theta_lat, n_R_lat)

        decode_mask_folded = mask_folded[:B].repeat(steps, 1, 1, 1, 1)
        recon_fields_folded = self.vqvae.decode_from_indices(gen_indices_folded, decode_mask_folded)

        return recon_fields_folded.view(B, steps, C, nz, n_theta, n_R)

    @torch.no_grad()
    def forward_from_geo(
        self,
        x_init_field,
        datetimes,
        poro_field,
        perm_field,
        spatial_mask,
        maxR,
        total_z_depth,
        steps=1,
        dynamic_features=None
    ):
        B, nt_init = x_init_field.shape[:2]

        _, static_features, dynamic_features, feature_mask = self.build_features_from_geo(
            poro_field=poro_field,
            perm_field=perm_field,
            spatial_mask=spatial_mask,
            nt_init=nt_init,
            dynamic_features=dynamic_features
        )

        return self.forward(
            x_init_field=x_init_field,
            datetimes=datetimes,
            static_features=static_features,
            dynamic_features=dynamic_features,
            spatial_mask=spatial_mask,
            feature_mask=feature_mask,
            maxR=maxR,
            total_z_depth=total_z_depth,
            steps=steps
        )

    @torch.no_grad()
    def forward_from_latent(
        self,
        x_init_field,
        datetimes,
        z_poro,
        z_perm,
        latent_mask,
        spatial_mask,
        maxR,
        total_z_depth,
        steps=1,
        dynamic_features=None
    ):
        B, nt_init = x_init_field.shape[:2]

        _, static_features, dynamic_features, feature_mask = self.build_features_from_latent(
            z_poro=z_poro,
            z_perm=z_perm,
            latent_mask=latent_mask,
            nt_init=nt_init,
            dynamic_features=dynamic_features
        )

        return self.forward(
            x_init_field=x_init_field,
            datetimes=datetimes,
            static_features=static_features,
            dynamic_features=dynamic_features,
            spatial_mask=spatial_mask,
            feature_mask=feature_mask,
            maxR=maxR,
            total_z_depth=total_z_depth,
            steps=steps
        )

    # =========================================================
    # 4. Training
    # =========================================================
    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        支持三种训练输入形式：

        旧版模式 (8项):
            (
                window_fields, datetimes, features,
                spatial_mask_orig, feature_mask, hard_data_locs,
                maxR, total_z_depth
            )

        在线 GeoVAE 模式 (9项):
            (
                window_fields, datetimes,
                poro_field, perm_field,
                dynamic_features,
                spatial_mask_orig,
                hard_data_locs,
                maxR, total_z_depth
            )

        预提 latent 模式 (10项):
            (
                window_fields, datetimes,
                z_poro, z_perm, latent_mask,
                dynamic_features,
                spatial_mask_orig,
                hard_data_locs,
                maxR, total_z_depth
            )
        """
        if len(batch) == 8:
            window_fields, datetimes, features, spatial_mask_orig, feature_mask, hard_data_locs, maxR, total_z_depth = batch

        elif len(batch) == 9:
            window_fields, datetimes, poro_field, perm_field, dynamic_features, spatial_mask_orig, hard_data_locs, maxR, total_z_depth = batch
            B, T = window_fields.shape[:2]
            features, _, _, feature_mask = self.build_features_from_geo(
                poro_field=poro_field,
                perm_field=perm_field,
                spatial_mask=spatial_mask_orig,
                nt_init=T,
                dynamic_features=dynamic_features
            )

        elif len(batch) == 10:
            window_fields, datetimes, z_poro, z_perm, latent_mask, dynamic_features, spatial_mask_orig, hard_data_locs, maxR, total_z_depth = batch
            B, T = window_fields.shape[:2]
            features, _, _, feature_mask = self.build_features_from_latent(
                z_poro=z_poro,
                z_perm=z_perm,
                latent_mask=latent_mask,
                nt_init=T,
                dynamic_features=dynamic_features
            )

        else:
            raise ValueError(
                f"training_step 期望 batch 长度为 8 / 9 / 10，但实际收到 {len(batch)}"
            )

        B, T, C, nz, n_theta, n_R = window_fields.shape

        x_folded = window_fields.reshape(B * T, C, nz, n_theta, n_R)
        mask_folded = spatial_mask_orig.unsqueeze(1).unsqueeze(1).expand(
            B, T, 1, nz, n_theta, n_R
        ).reshape(B * T, 1, nz, n_theta, n_R)

        with torch.no_grad():
            indices, _, _ = self.vqvae.encode_to_indices(x_folded, mask_folded)

        nz_lat, n_theta_lat, n_R_lat = indices.shape[1:]
        Ld = nz_lat * n_theta_lat * n_R_lat
        window_tokens = indices.view(B, T, Ld)
        latent_spatial_mask = (window_tokens[:, 0, :] != 0).bool()

        if features.shape[2] != Ld:
            raise ValueError(
                f"训练阶段 features 的 L_src={features.shape[2]} 与动态 token Ld={Ld} 不一致。"
            )

        lm_loss, logits = self.transformer._training_step(
            window_tokens,
            datetimes,
            features,
            latent_spatial_mask,
            feature_mask,
            maxR,
            total_z_depth
        )

        self.log('train/lm_loss', lm_loss, prog_bar=True)

        if self.freeze_vqvae:
            loss = lm_loss
        else:
            grid_dims = (B, T, C, nz, n_theta, n_R)
            latent_dims = (nz_lat, n_theta_lat, n_R_lat)

            joint_physics_loss = self._compute_joint_physics_loss(
                logits, window_fields, mask_folded, hard_data_locs, grid_dims, latent_dims
            )
            loss = lm_loss + joint_physics_loss

        self.log('train_pipeline_loss', loss, prog_bar=True)
        return loss

    def _compute_joint_physics_loss(
        self,
        logits,
        window_fields,
        mask_folded,
        hard_data_locs,
        grid_dims,
        latent_dims
    ) -> torch.Tensor:
        B, T, C, nz, n_theta, n_R = grid_dims
        nz_lat, n_theta_lat, n_R_lat = latent_dims

        probs = F.softmax(logits, dim=-1)
        weight_norm = F.normalize(self.vqvae.vq_layer.embedding.weight, p=2, dim=1)
        z_q_pred_continuous = torch.matmul(probs, weight_norm)

        z_q_pred_continuous = z_q_pred_continuous.view(
            B * (T - 1), nz_lat, n_theta_lat, n_R_lat, self.vqvae.hparams.latent_dim
        )
        z_q_pred_continuous = z_q_pred_continuous.permute(0, 4, 1, 2, 3).contiguous()

        target_mask_folded = (
            mask_folded.view(B, T, 1, nz, n_theta, n_R)[:, 1:, ...]
            .reshape(B * (T - 1), 1, nz, n_theta, n_R)
        )

        x_recon_pred = self.vqvae.decoder(z_q_pred_continuous, target_mask_folded)
        x_target = window_fields[:, 1:, ...].reshape(B * (T - 1), C, nz, n_theta, n_R)

        recons_loss = F.l1_loss(x_recon_pred, x_target, reduction='none')
        recons_loss = (recons_loss * target_mask_folded).sum() / (target_mask_folded.sum() + 1e-8)
        self.log('train/recons_loss', recons_loss)

        physical_hd_loss = 0.0
        if hard_data_locs is not None:
            if hard_data_locs.shape[0] == B:
                hard_data_locs_folded = (
                    hard_data_locs.unsqueeze(1)
                    .expand(-1, T - 1, -1, -1)
                    .reshape(B * (T - 1), -1, hard_data_locs.shape[-1])
                )
            else:
                hard_data_locs_folded = hard_data_locs

            hd_out = self.vqvae._compute_hd_loss(
                x_recon_pred, x_target, target_mask_folded, hard_data_locs_folded
            )

            if isinstance(hd_out, tuple):
                physical_hd_loss, hd_loss_dict = hd_out
                for key, val in hd_loss_dict.items():
                    self.log(f"train/joint_hd_{key}", val)
            else:
                physical_hd_loss = hd_out

            self.log('train/hd_loss', physical_hd_loss)

        return 1.0 * recons_loss + 10.0 * physical_hd_loss

    def configure_optimizers(self):
        if self.freeze_vqvae:
            param_groups = [{'params': self.transformer.parameters(), 'lr': 1e-4}]

            if self.use_geo_static:
                static_proj_params = list(self.poro_proj.parameters()) + list(self.perm_proj.parameters())
                # 🌟 新增：把动态特征投影层的参数也加进优化器！
                static_proj_params += list(self.dynamic_proj.parameters())
                param_groups.append({'params': static_proj_params, 'lr': 1e-4})

            return torch.optim.Adam(param_groups)
        else:
            param_groups = [
                {'params': self.transformer.parameters(), 'lr': 1e-5},
                {'params': self.vqvae.parameters(), 'lr': 1e-6}
            ]

            if self.use_geo_static:
                static_proj_params = list(self.poro_proj.parameters()) + list(self.perm_proj.parameters())
                # 🌟 新增：把动态特征投影层的参数也加进优化器！
                static_proj_params += list(self.dynamic_proj.parameters())
                param_groups.append({'params': static_proj_params, 'lr': 1e-4})

            return torch.optim.Adam(param_groups)


# ==========================================
# 🚀 使用 static_latents.pt 的本地测试
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 启动 LatentDynamicsPipeline(static_latents.pt) 整合测试...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前运行设备: {device}")

    # ---------------- 模拟组件初始化 ----------------
    # 注意：这里的 VQ-VAE latent 网格必须与你的 static_latents.pt 对齐
    nz_orig, n_theta_orig, n_R_orig = 20, 20, 10
    nz_lat, n_theta_lat, n_R_lat = 10, 10, 5   # 这里只是示例
    Ld_lat = nz_lat * n_theta_lat * n_R_lat

    vqvae = VQVAE3D(
        in_channels=5,
        hidden_dim=32,
        latent_dim=64,
        num_embeddings=1024,
        commitment_cost=0.25,
        num_res_blocks=2,
        num_attn_heads=8,
        compress_shape=(2, 2, 2)
    ).to(device)

    transformer = DecoderFusionTransformer(
        vocab_size=1025,
        d_model=128,
        num_heads=4,
        head_splits=[4, 2, 8],
        num_layers=2,
        hidden_dim=256,
        max_nt_width=10,
        Ld=Ld_lat,
        grid_shape=(nz_lat, n_theta_lat, n_R_lat),
    ).to(device)

    from vae.geo.geo_vae import PoroVAE3D, PermVAE3D

    poro_vae = PoroVAE3D(poro_min=0.0, poro_max=0.4, latent_channels=4).to(device)
    perm_vae = PermVAE3D(perm_min=0.1, perm_max=10000.0, latent_channels=8).to(device)

    pipeline = LatentDynamicsPipeline(
        vqvae_model=vqvae,
        transformer_model=transformer,
        poro_vae=poro_vae,
        perm_vae=perm_vae,
        d_model=128,
        freeze_vqvae=True,
        freeze_static_vae=True,
        dummy_dynamic_groups=1
    ).to(device)

    # ---------------- 模拟动态输入 ----------------
    B = 2
    nt_width = 4
    T = nt_width + 1
    C = 5
    maxR, total_z_depth = 1000.0, 50.0

    window_fields_raw = torch.randn(B, T, C, nz_orig, n_theta_orig, n_R_orig, device=device)
    datetimes = torch.arange(nt_width, device=device).float()
    spatial_mask_orig = (torch.rand(B, nz_orig, n_theta_orig, n_R_orig, device=device) > 0.2)
    hard_data_locs = None
    dynamic_features = None

    # ---------------- 读取 static_latents.pt ----------------
    latent_data = torch.load("E:/jbgs-model-training-main(5)/vae/geo/static_latents_10x10x5.pt", map_location=device)

    z_poro = latent_data["z_poro"].to(device).float()
    z_perm = latent_data["z_perm"].to(device).float()
    latent_mask = latent_data["latent_mask_poro"].to(device).float()

    pipeline.log = lambda *args, **kwargs: None

    print("\n⏳ 开始 static_latents.pt 训练测试...")

    # 预提 latent 模式 batch (10项)
    batch = (
        window_fields_raw,
        datetimes,
        z_poro,
        z_perm,
        latent_mask,
        dynamic_features,
        spatial_mask_orig,
        hard_data_locs,
        maxR,
        total_z_depth
    )

    try:
        loss = pipeline.training_step(batch, batch_idx=0)
        print("✅ training_step 通过")
        print(f"🎯 输出联合 Loss: {loss.item():.4f}")
    except Exception as e:
        print("❌ training_step 失败")
        print(e)

    try:
        pred = pipeline.forward_from_latent(
            x_init_field=window_fields_raw[:, :nt_width],
            datetimes=datetimes[:nt_width],
            z_poro=z_poro,
            z_perm=z_perm,
            latent_mask=latent_mask,
            spatial_mask=spatial_mask_orig,
            maxR=maxR,
            total_z_depth=total_z_depth,
            steps=1,
            dynamic_features=None
        )
        print("✅ forward_from_latent 通过")
        print(f"预测输出形状: {tuple(pred.shape)}")
    except Exception as e:
        print("❌ forward_from_latent 失败")
        print(e)

    if torch.cuda.is_available():
        print(f"📊 Pipeline 整合显存占用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    print("=" * 60)