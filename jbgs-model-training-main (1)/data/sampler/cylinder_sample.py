import torch
import numpy as np
import math
import random
from scipy.spatial import cKDTree
from pathlib import Path
import torch.nn.functional as F
from data.interfaces import IModelData, IWellData, IWellSmry

# =============================================================================
# 1. 极致优化的底层极坐标采样器 (状态持有者)
# =============================================================================
class CylinderSampler:
    """
    极速极坐标采样引擎。
    在 __init__ 中一次性预计算所有 KD-Tree、极坐标网格几何与物理衰减权重。
    调用 iter_cylinder 时，仅执行纯粹的张量切片和 Numba 级别的数值运算。
    """
    def __init__(self, 
                 actnum_3d: np.ndarray, 
                 cx: np.ndarray, cy: np.ndarray, sz: float,
                 global_wells_dict: dict,
                 nθ: int, nR: int, model_nz: int, 
                 θ0: float = 0.0, maxR_factor: float = 1.5, max_hd_points: int = 20):
        
        self.eclipse_nz, self.ny, self.nx = actnum_3d.shape
        self.actnum_1d = actnum_3d.ravel()
        self.mask_3d = actnum_3d.reshape(self.eclipse_nz, self.ny * self.nx)
        
        self.nθ, self.nR, self.model_nz = nθ, nR, model_nz
        self.θ0 = θ0
        self.max_hd_points = max_hd_points
        self.global_wells_dict = global_wells_dict

        # --- 1. 预构建网格 KD-Tree 与面积参数 ---
        self.centers_2d = np.stack([cx.ravel(), cy.ravel()], axis=1)
        self.grid_tree = cKDTree(self.centers_2d)
        
        sx, sy = cx.max() - cx.min(), cy.max() - cy.min()
        area, self.avg_cell_area = sx * sy, (sx / self.nx) * (sy / self.ny)

        # --- 2. 预构建井网 KD-Tree (用于光速提取邻井 Hard Data) ---
        self.all_well_names = list(global_wells_dict.keys())
        all_well_coords = [info[2] for info in global_wells_dict.values()]
        self.well_tree = cKDTree(np.array(all_well_coords)) if all_well_coords else None

        # --- 3. 预计算极坐标几何与寻路边界 ---
        n_wells = len(global_wells_dict)
        if n_wells == 0:
            raise ValueError("没有找到任何有效的活井，采样器初始化失败。")
            
        self.maxR = maxR_factor * math.sqrt(area / (math.pi * n_wells))
        a = 2.0
        R_edges = self.maxR * (np.exp(a * np.linspace(0, 1, nR + 1)) - 1) / (np.exp(a) - 1)
        R_centers = (R_edges[:-1] + R_edges[1:]) / 2.0
        Delta_R = R_edges[1:] - R_edges[:-1]
        
        Delta_theta = 2 * math.pi / nθ
        theta_edges = np.linspace(0, 2 * math.pi, nθ + 1)
        theta_centers = theta_edges[:-1] + (Delta_theta / 2.0)
        
        T_C, R_C = np.meshgrid(theta_centers, R_centers, indexing='ij')
        
        self.rel_pos = np.stack([R_C * np.cos(T_C - θ0), R_C * np.sin(T_C - θ0)], axis=-1)
        
        max_polar_area = 0.5 * Delta_theta * (R_edges[-1]**2 - R_edges[-2]**2)
        self.K_search = max(1, min(100, int(np.ceil((max_polar_area / self.avg_cell_area) * 2.0))))

        # 寻路宽容度常数
        decay_factor = np.exp(-3.0 * (R_C / self.maxR))
        self.lambda_flat = (1.0 + 3.0 * decay_factor).reshape(-1, 1)
        self.alpha_flat = (1.0 + 3.0 * decay_factor).reshape(-1, 1)
        self.R_C_flat = R_C.reshape(-1, 1)
        self.T_C_flat = T_C.reshape(-1, 1)
        self.D_R_flat = np.broadcast_to(Delta_R, (nθ, nR)).reshape(-1, 1)
        self.Delta_theta = Delta_theta
        self.R_edges = R_edges

    def iter_cylinder(self, field_raw: np.ndarray, target_wells: list, visualize: bool = False):
        """核心发牌引擎：传入多通道属性和目标井名，产出 Tensor 数据流"""
        if field_raw.ndim == 1:
            field_raw = field_raw[np.newaxis, :]
        num_channels = field_raw.shape[0]

        # 重建 3D 物理场 Tensor (C, nz, N_act) -> (C, nz, ny*nx)
        field_3d = np.zeros((num_channels, self.eclipse_nz * self.ny * self.nx), dtype=np.float32)
        field_3d[:, self.actnum_1d] = field_raw
        field_3d = field_3d.reshape(num_channels, self.eclipse_nz, self.ny * self.nx)

        for w_name in target_wells:
            if w_name not in self.global_wells_dict: continue
            
            w_i, w_j, w_xy = self.global_wells_dict[w_name]
            
            # --- 寻路核心 ---
            abs_pos = w_xy + self.rel_pos
            distances, indices = self.grid_tree.query(abs_pos.reshape(-1, 2), k=self.K_search)

             # k=1 时的 distances 会降维，需要 reshape
            if self.K_search == 1:
                indices = indices.reshape(-1, 1)
                distances = distances.reshape(-1, 1)
            
            cand_pos = self.centers_2d[indices]
            dx, dy = cand_pos[..., 0] - w_xy[0], cand_pos[..., 1] - w_xy[1]
            r_cand, t_cand = np.hypot(dx, dy), (np.arctan2(dy, dx) - self.θ0) % (2 * math.pi)
            
            dr_diff = np.abs(r_cand - self.R_C_flat)
            dt_diff = np.abs((t_cand - self.T_C_flat + math.pi) % (2 * math.pi) - math.pi)
            
            valid_mask = (dr_diff <= self.lambda_flat * self.D_R_flat / 2.0) & \
                         (dt_diff <= self.alpha_flat * self.Delta_theta / 2.0)
            valid_mask[~valid_mask.any(axis=1), 0] = True 

            # --- Z 轴并行聚合 ---
            vals_3d = field_3d[:, :, indices.flatten()].reshape(num_channels, self.eclipse_nz, -1, self.K_search)
            act_3d = self.mask_3d[:, indices.flatten()].reshape(self.eclipse_nz, -1, self.K_search)
            
            final_mask = valid_mask[np.newaxis, :, :] & act_3d
            count = final_mask.sum(axis=2)
            count_C = np.maximum(count[np.newaxis, ...], 1)
            final_mask_C = final_mask[np.newaxis, ...]

            mu = np.where(final_mask_C, vals_3d, 0.0).sum(axis=3) / count_C
            sum_sq = np.where(final_mask_C, vals_3d**2, 0.0).sum(axis=3)
            sigma = np.sqrt(np.maximum((sum_sq / count_C) - mu**2, 0))
            
            noise = np.random.normal(0, 1, size=(num_channels, self.eclipse_nz, self.nθ * self.nR)).astype(np.float32)
            sampled_vals = mu + sigma * noise
            
            count_broadcast = count[np.newaxis, ...]
            sampled_vals = np.where(count_broadcast <= 1, mu, sampled_vals)
            sampled_vals = np.where(count_broadcast == 0, 0.0, sampled_vals)
            
            field_t_raw = torch.from_numpy(sampled_vals.reshape(num_channels, self.eclipse_nz, self.nθ, self.nR))
            mask_t_raw = torch.from_numpy((count > 0).astype(np.float32).reshape(1, self.eclipse_nz, self.nθ, self.nR))

            # --- 硬数据 (邻井) 提取 ---
            hd_locs, hd_locs_names = [], [w_name]
            hd_locs_val_raw = torch.zeros((num_channels, self.eclipse_nz, self.nθ, self.nR), dtype=torch.float32)
            
            neighbor_indices = self.well_tree.query_ball_point(w_xy, self.maxR)
            for n_idx in neighbor_indices:
                other_name = self.all_well_names[n_idx]
                if other_name == w_name: continue
                
                other_i, other_j, other_xy = self.global_wells_dict[other_name]
                other_flat = other_j * self.nx + other_i
                
                dx, dy = other_xy[0] - w_xy[0], other_xy[1] - w_xy[1]
                r, theta = math.hypot(dx, dy), (math.atan2(dy, dx) - self.θ0) % (2 * math.pi)
                
                idx_r = np.clip(np.searchsorted(self.R_edges, r) - 1, 0, self.nR - 1)
                idx_t = int((theta / (2 * math.pi)) * self.nθ) % self.nθ
                
                hd_locs.append([1.0, float(idx_t), float(idx_r)])
                hd_locs_names.append(other_name)
                
                hd_z_vals = field_3d[:, :, other_flat]
                hd_z_mask = self.mask_3d[:, other_flat]
                for c in range(num_channels):
                    hd_locs_val_raw[c, :, idx_t, idx_r] = torch.from_numpy(np.where(hd_z_mask, hd_z_vals[c], 0)).float()

            if len(hd_locs) > self.max_hd_points:
                hd_locs = hd_locs[:self.max_hd_points]
            else:
                while len(hd_locs) < self.max_hd_points: hd_locs.append([0.0, 0.0, 0.0])
            hd_locs_t = torch.tensor(hd_locs, dtype=torch.float32)

            # --- Z 轴统一插值降维 ---
            field_t_raw, mask_t_raw, hd_locs_val_raw = field_t_raw.unsqueeze(0), mask_t_raw.unsqueeze(0), hd_locs_val_raw.unsqueeze(0)

            if self.eclipse_nz != self.model_nz:
                field_t = F.interpolate(field_t_raw, size=(self.model_nz, self.nθ, self.nR), mode='trilinear', align_corners=True)
                mask_t = F.interpolate(mask_t_raw, size=(self.model_nz, self.nθ, self.nR), mode='nearest')
                hd_locs_val_t = F.interpolate(hd_locs_val_raw, size=(self.model_nz, self.nθ, self.nR), mode='trilinear', align_corners=True)
            else:
                field_t, mask_t, hd_locs_val_t = field_t_raw, mask_t_raw, hd_locs_val_raw

            sample = {
                "well_name": w_name,
                "field_data": field_t.squeeze(0),
                "mask": (mask_t.squeeze(0) > 0.5).bool(),
                "hd_locs": hd_locs_t,
                "hd_locs_val": hd_locs_val_t.squeeze(0),
                "r_edges": self.R_edges,
                "n_theta": self.nθ,
                "model_nz": self.model_nz
            }

            if visualize:
                sample["_visual_req"] = {
                    "valid_2d_indices": indices[valid_mask],
                    "hd_well_names": hd_locs_names
                }
            yield sample

# =============================================================================
# 2. 宏观发牌人 (集成了 Train/Val 优雅切分)
# =============================================================================
class EclipseDataSampler:
    """
    负责单项目生命周期管理、死井剔除、固定 Seed 切分 Train/Val 集合，
    并向 Dataset 暴露优雅的流式接口。
    """
    def __init__(self, project_dir: str, project_name: str, 
                 data_source_type: str = 'eclipse',
                 sch_file=None, ev_file=None, vol_files=None, trj_file=None,
                 nθ: int = 26, nR: int = 10, model_nz: int = 40, 
                 train_ratio: float = 0.8, seed: int = 42):
        
        # 🌟 简单工厂模式：根据数据源路由，对下游保持极简
        if data_source_type == 'eclipse':
            from data.eclipse.model_data import EclipseModelData
            from data.eclipse.well_data import EclipseWellData
            from data.eclipse.well_smry import EclipseWellSmry
            
            self.model: IModelData = EclipseModelData(project_dir, project_name)
            self.well_data: IWellData = EclipseWellData(
                project_dir, project_name,
                sch_file=sch_file, ev_file=ev_file, vol_files=vol_files, trj_file=trj_file
            )
            
            self.well_smry: IWellSmry = EclipseWellSmry(project_dir, project_name)

                
        elif data_source_type == 'zarr':
            from data.zarr.zarr_data import ZarrModelData, ZarrWellData, ZarrWellSmry
            
            # Zarr 模式下，project_dir 实际上应该传入 .zarr 文件夹的路径
            self.model: IModelData = ZarrModelData(project_dir, project_name)
            self.well_data: IWellData = ZarrWellData(project_dir, project_name)
            self.well_smry: IWellSmry = ZarrWellSmry(project_dir, project_name)
            
        else:
            raise ValueError(f"不支持的数据源类型: {data_source_type}")
        
        # 1. 提取静态体系 (需要确保 model_data.py 中有 get_top_2d_centers)
        self.eclipse_nz, self.ny, self.nx = self.model.get_3Dframe_dim()
        self.actnum_3d = self.model.get_model_actnum()
        self.cx, self.cy = self.model.get_top_2d_centers()
        self.z_size, _, _ = self.model.get_3Dframe_size()

        # 2. 全局死井剔除 (一次性构建纯净的活井字典)
        snapshot_df = self.well_data.get_steps_info(-1).drop_duplicates(subset=['well'], keep='last')
        self.global_wells_dict = {}
        for _, row in snapshot_df.iterrows():
            w_name = row['well']
            comps = self.well_data.static_wells.get(w_name, {}).get("completions", [])
            is_active = False
            for comp in comps:
                c_i, c_j, c_k = comp["I"] - 1, comp["J"] - 1, comp["K"] - 1
                if 0 <= c_k < self.eclipse_nz and 0 <= c_j < self.ny and 0 <= c_i < self.nx:
                    if self.actnum_3d[c_k, c_j, c_i]:
                        is_active = True; break
            if is_active and comps:
                w_j, w_i = comps[0]["J"] - 1, comps[0]["I"] - 1
                self.global_wells_dict[w_name] = (w_i, w_j, np.array([self.cx[w_j, w_i], self.cy[w_j, w_i]]))

        # 3. 🌟 优雅的 Train/Val 空间切分
        sorted_wells = sorted(self.global_wells_dict.keys())
        rng = random.Random(seed)
        rng.shuffle(sorted_wells)
        split_idx = max(1, int(len(sorted_wells) * train_ratio))
        
        self.split_wells = {
            "train": sorted_wells[:split_idx],
            "val": sorted_wells[split_idx:],
            "all": sorted_wells
        }

        # 4. 初始化核心纯数学引擎
        self.cylinder_sampler = CylinderSampler(
            self.actnum_3d, self.cx, self.cy, self.z_size / self.eclipse_nz,
            self.global_wells_dict, nθ, nR, model_nz
        )

    def _get_target_wells(self, mode: str, shuffle: bool):
        targets = self.split_wells.get(mode, self.split_wells["all"]).copy()
        if shuffle and mode == "train": # 验证集通常不需要打乱
            np.random.shuffle(targets)
        return targets

    def iter_static(self, field_name: str, mode: str = "train", 
                    visualize: bool = False, shuffle: bool = True):
        targets = self._get_target_wells(mode, shuffle)
        if not targets: return
            
        field_raw = self.model.get_static_fields([field_name])

        for sample in self.cylinder_sampler.iter_cylinder(field_raw, targets, visualize):
            sample["field_name"] = field_name
            sample["time_info"] = "Static"

            yield sample

    def iter_dynamic(self, field_names: list, mode: str = "train", 
                     visualize: bool = False, shuffle: bool = True, target_steps: list = None):
        
        # 1. 获取模型每一步的绝对时间 (通货)
        model_dt_map = self.model.get_dynamic_datetimes()

        if target_steps is None:
            model_steps = self.model.get_dynamic_steps()
        else:
            model_steps = target_steps
        
        if not model_steps: return
            
        # 过滤掉无法解析时间的死步
        valid_steps = [s for s in model_steps if s in model_dt_map]

        target_dts = [model_dt_map[s] for s in valid_steps]
        
        base_targets = self._get_target_wells(mode, shuffle=False)
        if not base_targets: return

   
        # ==========================================================
        # 🌟 极简交割：直接拿 datetime_steps 问井数据要人
        # ==========================================================
        dt_to_wells = self.well_data.get_step_to_active_wells_map(
            target_steps=target_dts, 
            target_wells=base_targets
        )

        # --- 【时间步级别】的全局 Shuffle ---
        if shuffle and mode == "train":
            np.random.shuffle(valid_steps)

        # --- 极速发牌双重循环 ---
        for step in valid_steps:
            dt = model_dt_map[step]               
            target_wells = dt_to_wells.get(dt, []) 
            
            if not target_wells: 
                continue
                
            if shuffle and mode == "train":
                np.random.shuffle(target_wells)
                
            # 拿到这一步的真实全区内存数据
            field_raw_multi = self.model.get_dynamic_fields(step, field_names)
            active_wells_set = set(target_wells)
            
            for sample in self.cylinder_sampler.iter_cylinder(field_raw_multi, target_wells, visualize):
                sample["step_idx"] = step
                sample["field_name"] = field_names[0] if isinstance(field_names, list) else field_names
                sample["time_info"] = f"Step {step} ({dt.strftime('%Y-%m-%d')})"
                
                yield sample

# =============================================================================
# 3. 外部测试渲染器 (仅消费 Numpy，零 PyVista 依赖)
# =============================================================================
if __name__ == "__main__":

    import pyvista as pv
    from data.eclipse.model_data import build_pv_faces_numba

    def debug_render(sample_dict: dict, origin: tuple):
        if "visual_items" not in sample_dict:
            print("未开启 visualize=True，无渲染资产。")
            return
            
        v_items = sample_dict["visual_items"]
        plotter = pv.Plotter()
        
        quads = v_items["quads"]
        if quads.size > 0:
            quads_centered = quads - np.array(origin)
            verts = np.ascontiguousarray(quads_centered.reshape(-1, 3), dtype=np.float64)
            faces_1d = build_pv_faces_numba(len(quads))
            mesh = pv.PolyData(verts, faces_1d)
            plotter.add_mesh(mesh, cmap='jet', show_edges=True, opacity=0.9)

        for track in v_items["tracks"]:
            color = track["color"]
            pts = track["track_coords"] - np.array(origin)
            lines = np.array([len(pts)] + list(range(len(pts))), dtype=np.int64)
            plotter.add_mesh(pv.PolyData(pts, lines=lines), color=color, line_width=4.0, render_lines_as_tubes=True)
            
            perf_pts = track["perf_coords"] - np.array(origin)
            plotter.add_mesh(pv.PolyData(perf_pts), color=color, point_size=10.0, render_points_as_spheres=True)
            
            lbl_pt = track["label_coord"] - np.array(origin)
            lbl_pt[2] += 30
            plotter.add_point_labels(lbl_pt[np.newaxis, :], [track["well"]], text_color=color, shape_opacity=0.3)

        plotter.set_background('white')
        plotter.show()


if __name__ == "__main__":
    import time
    PROJECT_DIR, PROJECT_NAME = r"D:\\YihengZhu\\jbgs-lma\\proj-phase1\\eclipse-models\\1c\\water_flooding", "BYEPD93"
    
    print("="*60)
    print("🚀 启动流式极坐标发牌人性能测试...")
    
    # ---------------------------------------------------------
    # 测试 1: 系统初始化性能
    # ---------------------------------------------------------
    t0 = time.time()
    dealer = EclipseDataSampler(PROJECT_DIR, PROJECT_NAME, seed=42)
    t_init = time.time() - t0
    
    print(f"\n✅ [1. 初始化阶段] 耗时: {t_init:.3f} 秒")
    print(f"   -> KD-Tree构建、面积计算、死井剔除、Train/Val 空间切分已完成")
    print(f"   -> 训练井: {dealer.split_wells['train']}")
    print(f"   -> 验证井: {dealer.split_wells['val']}")
    
    # ---------------------------------------------------------
    # 测试 2: 静态场抽取性能 (单次)
    # ---------------------------------------------------------
    t0 = time.time()
    static_gen = dealer.iter_static("PORO", mode="train", visualize=True)
    sample_static = next(static_gen)
    t_static = time.time() - t0
    
    print(f"\n✅ [2. 静态发牌阶段] 单张牌抽取耗时: {t_static:.3f} 秒")
    print(f"   -> 获取到: {sample_static['well_name']} 的 PORO 场")
    print(f"   -> Tensor 形状: {sample_static['field_data'].shape}")
    
    # ---------------------------------------------------------
    # 测试 3: 动态联合场抽取性能 (首次冷启动 I/O + VAE 数据流)
    # ---------------------------------------------------------
    t0 = time.time()
    # 模拟 VAE 的 5 通道输入
    dynamic_gen = dealer.iter_dynamic(["PRESSURE", "SWAT"], mode="train", visualize=False)
    
    # 测算连续抽出 10 张牌的平均时间 (包含一次换时间步的 I/O)
    samples_drawn = 0
    t_io_start = time.time()
    
    for i, sample in enumerate(dynamic_gen):
        samples_drawn += 1
        if i == 0:
            t_first_dynamic = time.time() - t_io_start
            print(f"\n✅ [3. 动态发牌阶段] 首次抽取 (含硬盘 I/O): {t_first_dynamic:.3f} 秒")
            print(f"   -> 获取到: {sample['well_name']} @ Step {sample['step_idx']}")
            print(f"   -> Tensor 形状: {sample['field_data'].shape} (双通道)")
            
        if samples_drawn >= 10:
            break
            
    t_10_dynamic = time.time() - t0
    print(f"   -> 连续抽取 10 张动态牌总耗时: {t_10_dynamic:.3f} 秒 (平均 {t_10_dynamic/10:.4f} 秒/张)")

    # ---------------------------------------------------------
    # 可选：渲染最后一张牌验证物理几何
    # ---------------------------------------------------------
    debug_render(sample_static, dealer.model.get_origin())
    print("="*60)