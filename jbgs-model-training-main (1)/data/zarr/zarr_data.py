# data/zarr/zarr_data.py
import zarr
import pickle
import bisect
import numpy as np
import pandas as pd
import pyvista as pv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union, Any

from data.interfaces import IModelData, IWellData, IWellSmry
from data.numba_utils import (
    compute_bounds_numba, compute_cell_centers_numba, 
    get_boundary_quads_numba, build_pv_faces_numba, 
    map_sub_active_field_numba,
    build_act_remap, clean_nans_numba
)

class ZarrModelData(IModelData):
    """Zarr 极速模型数据提供者 (0文本解析，纯内存切片)"""
    def __init__(self, project_dir: str | Path, project_name: str):
        self.project_dir = Path(project_dir)
        self.project_name = project_name
        self.zarr_dir = self.project_dir / f"{self.project_name}.zarr"
        self.root = zarr.open(str(self.zarr_dir), mode='r')
        
        with open(self.zarr_dir / 'meta_payload.pkl', 'rb') as f:
            self.meta = pickle.load(f)['model']
            
        # 缓存核心几何张量
        self.coord_1d = self.root['coord_1d'][:]
        self.zcorn_1d = self.root['zcorn_1d'][:]
        self.base_actnum = self.root['actnum_3d'][:]
        
        self.nz, self.ny, self.nx = self.meta['frame_dim']
        self.nblocks = self.meta['nblocks']
        self.radial = self.meta['radial']
        
    def get_3Dframe_dim(self) -> Tuple[int, int, int]: return self.meta['frame_dim']

    def get_3Dframe_size(self, certain_actnum=None) -> Tuple[float, float, float]:
        if certain_actnum is None: return self.meta['frame_size']
        quads, _ = self.get_surface_quads(certain_actnum)
        mins, maxs = compute_bounds_numba(quads)
        sz, sy, sx = maxs - mins
        return (sz, sy, sx)
    
    def get_origin(self, certain_actnum=None) -> Tuple[float, float, float]: 
        if certain_actnum is None: return self.meta['origin']
        # 动态计算局部包围盒
        quads, _ = self.get_surface_quads(certain_actnum)
        mins, maxs = compute_bounds_numba(quads)
        return tuple((mins + maxs) / 2.0)
    
    def get_model_actnum(self) -> np.ndarray: return self.base_actnum.copy()
    def get_dynamic_steps(self) -> List[int]: return self.meta['dynamic_steps']
    def get_dynamic_datetimes(self) -> Dict[int, datetime]: return self.meta['dynamic_datetimes']
    
    def get_top_2d_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        pillars_per_block = (self.ny + 1) * (self.nx + 1)
        coord_block = self.coord_1d[:pillars_per_block * 6].reshape(self.ny + 1, self.nx + 1, 6)
        top_x, top_y = coord_block[..., 0], coord_block[..., 1]
        cx = (top_x[:-1, :-1] + top_x[1:, :-1] + top_x[:-1, 1:] + top_x[1:, 1:]) / 4.0
        cy = (top_y[:-1, :-1] + top_y[1:, :-1] + top_y[:-1, 1:] + top_y[1:, 1:]) / 4.0
        return cx, cy

    def get_cell_centers(self, I_list: List[int], J_list: List[int], K_list: List[int]) -> np.ndarray:
        return compute_cell_centers_numba(
            np.asarray(I_list, dtype=np.int64), np.asarray(J_list, dtype=np.int64), np.asarray(K_list, dtype=np.int64),
            self.coord_1d, self.zcorn_1d, self.nx, self.ny, self.nz, self.nblocks, self.radial, 1e-12
        )

    def get_static_fields(self, field_names: List[str], certain_actnum=None) -> np.ndarray:
        res = [self.root['static'][f][:] for f in field_names]
        arr = np.stack(res, axis=0) if len(res) > 1 else res[0]
        
        if certain_actnum is None: return arr
        
        # 🌟 修复：应用局部掩码映射
        target_size = np.count_nonzero(certain_actnum)
        if arr.ndim == 1:
            return map_sub_active_field_numba(self.base_actnum.ravel(), certain_actnum.ravel(), arr, target_size)
        else:
            return np.stack([map_sub_active_field_numba(self.base_actnum.ravel(), certain_actnum.ravel(), a, target_size) for a in arr])

    def get_dynamic_fields(self, idx_step: int, field_names: List[str], certain_actnum=None) -> np.ndarray:
        res = [self.root['dynamic'][f"Step_{idx_step}"][f][:] for f in field_names]
        arr = np.stack(res, axis=0) if len(res) > 1 else res[0]
        
        if certain_actnum is None: return arr
        
        target_size = np.count_nonzero(certain_actnum)
        if arr.ndim == 1:
            return map_sub_active_field_numba(self.base_actnum.ravel(), certain_actnum.ravel(), arr, target_size)
        else:
            return np.stack([map_sub_active_field_numba(self.base_actnum.ravel(), certain_actnum.ravel(), a, target_size) for a in arr])

    # --- 以下是兼容 3D 渲染所需的方法拷贝 (保证契约完整) ---
    def get_surface_quads(self, certain_actnum=None) -> Tuple[np.ndarray, np.ndarray]:
        actnum = self.base_actnum if certain_actnum is None else certain_actnum
        act_remap = build_act_remap(actnum, self.nx, self.ny, self.nz)
        return get_boundary_quads_numba(self.coord_1d, self.zcorn_1d, actnum, act_remap,
                                        self.nx, self.ny, self.nz, self.nblocks, self.radial, 1e-12)

    def get_pv_static_mesh(self, field_names: Union[str, List[str]], certain_actnum=None) -> Any:
        quads, quad_cells = self.get_surface_quads(certain_actnum)
        f_name = field_names[0] if isinstance(field_names, list) else field_names
        field_1d = self.get_static_fields([f_name], certain_actnum)
        return self._build_pv_mesh(quads, field_1d[quad_cells], f_name)

    def get_pv_dynamic_mesh(self, time_step: int, field_names: Union[str, List[str]], certain_actnum=None) -> Any:
        quads, quad_cells = self.get_surface_quads(certain_actnum)
        f_name = field_names[0] if isinstance(field_names, list) else field_names
        field_1d = self.get_dynamic_fields(time_step, [f_name], certain_actnum)
        return self._build_pv_mesh(quads, field_1d[quad_cells], f_name)

    def _build_pv_mesh(self, quads: np.ndarray, values: np.ndarray, field_name: str) -> Any:
        if quads.size == 0: return pv.PolyData()
        quads_centered = quads - np.array(self.get_origin())
        verts = np.ascontiguousarray(quads_centered.reshape(-1, 3), dtype=np.float64)
        mesh = pv.PolyData(verts, build_pv_faces_numba(len(quads)))
        mesh.cell_data[field_name] = clean_nans_numba(values)
        mesh.set_active_scalars(field_name)
        return mesh


class ZarrWellData(IWellData):
    """Zarr 井控数据替身 (内部 Pandas 逻辑复用)"""
    def __init__(self, project_dir: str | Path, project_name: str):
        self.project_dir = Path(project_dir)
        self.project_name = project_name
        self.zarr_dir = self.project_dir / f"{self.project_name}.zarr"

        with open(self.zarr_dir / 'meta_payload.pkl', 'rb') as f:
            meta = pickle.load(f)['well_data']
            
        self.dates = meta['dates']
        self._static_wells = meta['static_wells']
        self._trj_data = meta['trj_data']
        self.df_events = meta['df_events']

    @property
    def static_wells(self) -> dict: return self._static_wells

    @property
    def trj_data(self) -> dict: return self._trj_data

    def datetime_to_step_idx(self, target_dts: list) -> list:
        if not self.dates: return [0] * len(target_dts) 
        res, lo = [], 0
        for dt in target_dts:
            if dt is None:
                res.append(-1)
                continue
            idx = bisect.bisect_right(self.dates, dt, lo=lo)
            lo = idx 
            res.append(idx - 1 if idx > 0 else 0)
        return res

    def get_wells_by_step_idx_and_name(self, step_indices=None, well_names=None, cluster_by=None) -> pd.DataFrame:
        df_filtered = self.df_events
        if step_indices is not None:
            step_indices = [step_indices] if isinstance(step_indices, (int, datetime)) else step_indices
            target_indices = self.datetime_to_step_idx(step_indices) if isinstance(step_indices[0], datetime) else [i if i >= 0 else len(self.dates) + i for i in step_indices]
            if not target_indices: return pd.DataFrame()
            df_filtered = df_filtered[df_filtered['step_idx'] <= max(target_indices)]
            
        if well_names is not None:
            target_wells = [well_names] if isinstance(well_names, str) else well_names
            df_filtered = df_filtered[df_filtered['well'].isin(target_wells)]
            
        if df_filtered.empty: return pd.DataFrame()
        
        subset = df_filtered.copy().drop_duplicates(subset=['step_idx', 'well'], keep='last')
        cluster_by = cluster_by or ['step_idx', 'well']
        grouped = subset.groupby(cluster_by, sort=False)
        return subset.iloc[np.concatenate(list(grouped.indices.values()))].reset_index(drop=True)

    def get_steps_info(self, step_indices=None) -> pd.DataFrame:
        return self.get_wells_by_step_idx_and_name(step_indices=step_indices, cluster_by=['step_idx', 'well'])

    def get_step_to_active_wells_map(self, target_steps: list, target_wells: list = None) -> dict:
        if not target_steps: return {}
        base_targets = set(self.static_wells.keys()) if target_wells is None else set(target_wells)
        internal_steps = self.datetime_to_step_idx(target_steps) if isinstance(target_steps[0], datetime) else target_steps
        
        events_by_step = self.df_events.groupby('step_idx') if not self.df_events.empty else {}
        internal_map, current_active = {}, set()
        
        for step in range(max(internal_steps) + 1 if internal_steps else 0):
            if step in events_by_step.groups:
                for _, row in events_by_step.get_group(step).iterrows():
                    w = row['well']
                    if w in base_targets:
                        current_active.discard(w) if str(row.get('status', 'OPEN')).upper() == "SHUT" else current_active.add(w)
            if step in set(internal_steps):
                internal_map[step] = list(current_active)
                
        return {k: internal_map.get(v, []) for k, v in zip(target_steps, internal_steps)}

    def get_pv_well_tracks(self, model: IModelData, step_idx: int, well_names=None,
                           display_radius: float=3.0, extend_track_length: float=0.0,
                           show_perforation: bool=True, show_labels: bool=True, label_scale: int=14) -> Tuple[List[dict], List[dict]]:
        """纯复刻的渲染资产计算逻辑"""
        df_events = self.get_wells_by_step_idx_and_name(step_idx, well_names, cluster_by=['step_idx', 'well'])
        if df_events.empty: return [], []
        snapshot_df = df_events.drop_duplicates(subset=['well'], keep='last')
        
        target_wells = set(snapshot_df['well']).union(self.trj_data.keys() if well_names is None else [])
        color_groups = {}
        
        for well in target_wells:
            if well in snapshot_df['well'].values:
                row = snapshot_df[snapshot_df['well'] == well].iloc[0]
                status, keyword = str(row['status']).upper(), str(row['keyword']).upper()
            else:
                status, keyword = "OPEN", ""

            wtype = str(self.static_wells.get(well, {}).get("type", "")).upper()
            color = "gray" if status == "SHUT" else ("blue" if keyword == 'WCONINJE' or well.upper().startswith(('W','I','Z')) or "WAT" in wtype or "INJ" in wtype else "red")
            
            if color not in color_groups: color_groups[color] = {"track_coords": [], "perf_coords": [], "labels": [], "label_coords": []}
            track_coords, perf_coords = None, None

            if well in self.trj_data:
                raw_pts = self.trj_data[well].copy()
                raw_pts[:, 2] *= -1 
                track_coords, perf_coords = raw_pts, np.array([raw_pts[-1]])
            else:
                comps = self.static_wells.get(well, {}).get("completions", [])
                if not comps: continue
                try:
                    coords = model.get_cell_centers([c["I"] for c in comps], [c["J"] for c in comps], [c["K"] for c in comps])
                    perf_coords = coords.copy()
                    if extend_track_length > 0:
                        wellhead = coords[0].copy()
                        wellhead[2] += extend_track_length 
                        track_coords = np.vstack([wellhead, coords])
                    else: track_coords = coords
                except Exception: continue

            color_groups[color]["track_coords"].append(track_coords)
            if show_perforation: color_groups[color]["perf_coords"].append(perf_coords)
            color_groups[color]["labels"].append(well)
            color_groups[color]["label_coords"].append(track_coords[0])

        render_items, label_items = [], []
        for col, gdata in color_groups.items():
            if not gdata["track_coords"]: continue
            track_points = np.vstack(gdata["track_coords"])
            lines, offset = [], 0
            for c in gdata["track_coords"]:
                lines.extend([len(c)] + list(range(offset, offset + len(c))))
                offset += len(c)

            track_mesh = pv.PolyData(track_points)
            track_mesh.lines = np.array(lines, dtype=np.int64)
            render_items.append({"mesh": track_mesh, "kwargs": {"color": col, "line_width": display_radius, "render_lines_as_tubes": True, "lighting": True}})

            if show_perforation and gdata["perf_coords"]:
                render_items.append({"mesh": pv.PolyData(np.vstack(gdata["perf_coords"])), "kwargs": {"color": col, "point_size": display_radius * 1.5, "render_points_as_spheres": True, "lighting": True}})

            if show_labels and gdata["label_coords"]:
                lbl_pts = np.vstack(gdata["label_coords"])
                lbl_pts[:, 2] += 30
                label_items.append({"points": lbl_pts, "texts": gdata["labels"], "kwargs": {"text_color": col, "font_size": int(label_scale), "point_size": 0, "shape_opacity": 0.4, "shape": 'rounded_rect'}})

        return render_items, label_items


class ZarrWellSmry(IWellSmry):
    """Zarr 产量动态替身 (零解析，直取 DataFrame)"""
    def __init__(self, project_dir: str | Path, project_name: str):
        self.project_dir = Path(project_dir)
        self.project_name = project_name
        self.zarr_dir = self.project_dir / f"{self.project_name}.zarr"

        with open(self.zarr_dir / 'meta_payload.pkl', 'rb') as f:
            meta = pickle.load(f)['well_smry']
        self._df_field = meta['df_field']
        self._df_well = meta['df_well']

    @property
    def df_field(self) -> Optional[pd.DataFrame]: return self._df_field
    
    @property
    def df_well(self) -> Optional[pd.DataFrame]: return self._df_well