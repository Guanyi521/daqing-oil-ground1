# -*- coding: utf-8 -*-
import pyvista as pv
import os
import re
import bisect
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import numpy as np
from datetime import datetime
from opm.io.ecl import EGrid, EclFile, ERst
from opm.util import EModel
from data.numba_utils import (
    build_act_remap, 
    compute_bounds_numba,
    compute_cell_centers_numba,
    get_boundary_quads_numba,
    build_pv_faces_numba,
    map_sub_active_field_numba,
    clean_nans_numba
)
from data.interfaces import IModelData

# =============================================================================
# 核心数据抽象类
# =============================================================================

class EclipseModelData(IModelData):
    """
    Eclipse 静态与动态数据读取、网格坐标生成与 3D 渲染抽象类。
    """

    def __init__(self, project_dir: str | Path, project_name: str):
        """
        初始化解析器，加载基础网格文件和动态文件列表。
        
        Args:
            project_dir: 模型所在目录
            project_name: 模型主名称 (如 'BYEPD93')
        """
        self.project_dir = Path(project_dir)
        self.project_name = project_name
        
        # 1. 加载基础文件对象
        grid_path = self.project_dir / f"{project_name}.EGRID"
        init_path = self.project_dir / f"{project_name}.INIT"
        
        self.efile = EclFile(str(grid_path))
        self.egrid = EGrid(str(grid_path))
        self.emod = EModel(str(init_path))
        
        # 2. 获取基础网格维度
        self.nx, self.ny, self.nz = self._ecl_get(self.efile, "GRIDHEAD")[1:4]
        
        # 3. 提取全局有效网格布尔掩码
        actnum_raw = self._ecl_get(self.efile, "ACTNUM")
        self.base_actnum = actnum_raw.astype(bool).reshape(self.nz, self.ny, self.nx)
        self.base_active_cells = np.sum(self.base_actnum)
        
        # 4. 提取 COORD 与 ZCORN 原始数据 (一维化，float64)
        self.coord_1d = np.asarray(self._ecl_get(self.efile, "COORD"), dtype=np.float64).ravel()
        self.zcorn_1d = np.asarray(-self._ecl_get(self.efile, "ZCORN"), dtype=np.float64).ravel()
        
        
        # 5. 分析 COORD 块参数
        pillars_per_block = (self.ny + 1) * (self.nx + 1)
        block_size = pillars_per_block * 6
        self.nblocks = self.coord_1d.size // block_size
        if self.nblocks not in (1, self.nz):
            self.nblocks = 1
            
        coord_block = self.coord_1d[:block_size].reshape(self.ny + 1, self.nx + 1, 6)
        xt, yt = coord_block[..., 0], coord_block[..., 1]
        xb, yb = coord_block[..., 3], coord_block[..., 4]
        self.radial = bool(((np.abs(yt) <= 360) & (np.abs(yb) <= 360)).mean() > 0.98 and 
                           ((xt >= 0) & (xb >= 0)).mean() > 0.98)
        

        self.dynamic_files: Dict[int, Path] = None 

        self.update_dynamic_files()

        # 惰性缓存属性
        self._origin_cache: Optional[Tuple[float, float, float]] = None
        self._frame_size_cache: Optional[Tuple[float, float, float]] = None

    @staticmethod
    def _ecl_get(ecl_file: EclFile, target_kw: str) -> Optional[np.ndarray]:
        """内部工具：从 EclFile 中获取特定关键字的数据"""
        for index, kw in enumerate(ecl_file.arrays):
            if kw[0] == target_kw:
                return ecl_file[index]
        return None
    
    def _resolve_step(self, step: int | datetime) -> int:
        """统一动态时间步解析器，兼容 int, datetime 及负数倒数索引"""
        if isinstance(step, datetime):
            return self.datetime_to_step_idx([step])[0]
            
        if step < 0:
            steps = self.get_dynamic_steps()
            return steps[step] if steps else 0
            
        return step
    
    def update_dynamic_files(self):
        """ 扫描所有动态 .X 文件 """
        pattern = re.compile(f'^{re.escape(self.project_name)}\.X(\d+)$', re.IGNORECASE)
        files = []
        for f in os.listdir(self.project_dir):
            m = pattern.match(f)
            if m:
                files.append((int(m.group(1)), self.project_dir / f))
        files.sort(key=lambda x: x[0])
        self.dynamic_files = {step: path for step, path in files}


    def get_origin(self, certain_actnum: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        if self._origin_cache is None:
            quads, _ = self.get_surface_quads(certain_actnum)
            
            # [Optimized] O(1) 内存复杂度的边界计算
            mins, maxs = compute_bounds_numba(quads)
            
            self._origin_cache = tuple((mins + maxs) / 2.0)
            self._frame_size_cache = tuple(maxs - mins)
        return self._origin_cache

    def get_3Dframe_size(self, certain_actnum: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """
        返回外包络框的物理尺寸大小。
        
        Returns:
            tuple: (Z_size, Y_size, X_size) 的跨度长度
        """
        if self._frame_size_cache is None:
            self.get_origin(certain_actnum)  # 触发计算
        sx, sy, sz = self._frame_size_cache
        return (sz, sy, sx)

    def get_3Dframe_dim(self) -> Tuple[int, int, int]:
        """
        返回网格 3D 拓扑维度。
        
        Returns:
            tuple: (K=nZ, J=nY, I=nX)
        """
        return (self.nz, self.ny, self.nx)

    def get_model_actnum(self) -> np.ndarray:
        """
        返回当前模型的 3D 有效网格布尔数组。
        
        Returns:
            np.ndarray: shape为 (nz, ny, nx) 的 bool 数组
        """
        return self.base_actnum.copy()
    
    def get_top_2d_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        极速获取网格顶层 XY 投影面的 2D 中心矩阵。
        纯 NumPy 向量化操作，比严谨的 3D 中心点计算快几个数量级，适用于仅需平面拓扑的场景。
        """
        # 利用底层的连续内存切片，极速还原顶层坐标块
        pillars_per_block = (self.ny + 1) * (self.nx + 1)
        coord_block = self.coord_1d[:pillars_per_block * 6].reshape(self.ny + 1, self.nx + 1, 6)
        
        top_x, top_y = coord_block[..., 0], coord_block[..., 1]
        
        # 纯 C 级别向量化并行相加
        cx = (top_x[:-1, :-1] + top_x[1:, :-1] + top_x[:-1, 1:] + top_x[1:, 1:]) / 4.0
        cy = (top_y[:-1, :-1] + top_y[1:, :-1] + top_y[:-1, 1:] + top_y[1:, 1:]) / 4.0
        
        return cx, cy
    
    def get_cell_centers(self, I_list, J_list, K_list) -> np.ndarray:
        """
        批量获取网格(I, J, K)列表的物理中心坐标。
        
        Args:
            I_list, J_list, K_list: 1-based 的 Eclipse 原始索引列表或数组。
        Returns:
            np.ndarray: shape 为 (N, 3) 的三维坐标数组 [X, Y, Z]
        """
        I_arr = np.asarray(I_list, dtype=np.int64)
        J_arr = np.asarray(J_list, dtype=np.int64)
        K_arr = np.asarray(K_list, dtype=np.int64)
        
        return compute_cell_centers_numba(
            I_arr, J_arr, K_arr,
            self.coord_1d, self.zcorn_1d,
            self.nx, self.ny, self.nz, self.nblocks, self.radial, 1e-12
        )

    def get_surface_quads(self, certain_actnum: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算并返回外层暴露的网格表面四边形坐标(即 corner vertices)。
        
        Args:
            certain_actnum: 传入特定筛选的 active_cells 掩码, shape (nz, ny, nx)。
                            不传则默认使用全模型的 ACTNUM。
                            
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - quads: shape 为 (N_faces, 4, 3) 的外表面 4 顶点坐标
                - quad_cells: shape 为 (N_faces,) 的 1D 数组，代表每个 quad 所属的 active cell 的一维索引
        """
        actnum = self.base_actnum if certain_actnum is None else certain_actnum
        act_remap = build_act_remap(actnum, self.nx, self.ny, self.nz)
        
        quads_xyz, quad_cells = get_boundary_quads_numba(
            self.coord_1d, self.zcorn_1d, actnum, act_remap,
            self.nx, self.ny, self.nz, self.nblocks, self.radial, 1e-12
        )
        return quads_xyz, quad_cells

    def get_dynamic_steps(self) -> List[int]:
        """
        获取当前模型所有可用的动态时间步索引。
        
        Returns:
            List[int]: 从小到大排列的时间步列表 (如 [1, 2, 3, ...])
        """

        return list(self.dynamic_files.keys())

    def datetime_to_step_idx(self, target_dts: list) -> list:
        """
        汇率转换：找 >= datetime 的最近 model step_idx
        利用 Timsort O(N) 排序特性与 bisect 滑动窗口
        """
        dt_map = self.get_dynamic_datetimes() 
        
        # 🌟 Timsort 降维打击：因为天然有序，这里的 sorted 实际上是 O(N) 复杂度
        valid_items = sorted([(s, d) for s, d in dt_map.items() if d is not None], key=lambda x: x[1])
        if not valid_items:
            return [-1] * len(target_dts)
            
        steps = [x[0] for x in valid_items]
        dates = [x[1] for x in valid_items]
        
        res = []
        lo = 0  # 🌟 核心优化：滑动起点
        
        for dt in target_dts:
            if dt is None:
                res.append(-1)
                continue
                
            idx = bisect.bisect_left(dates, dt, lo=lo)
            lo = idx  # 🌟 更新起点
            
            if idx >= len(dates):
                res.append(steps[-1])
            else:
                res.append(steps[idx])
                
        return res
    
    def get_dynamic_days(self) -> Dict[int, float]:
        """
        从 Restart 文件的 DOUBHEAD 中提取物理模拟时间（累积天数）。
        
        Returns:
            Dict[int, float]: 格式为 {step_idx: time_in_days} 
        """
        step_day_map = {}
        
        for step_idx, file_path in self.dynamic_files.items():
            try:
                ecl_file = EclFile(str(file_path))
                doubhead = self._ecl_get(ecl_file, "DOUBHEAD")
                
                # Eclipse 标准：DOUBHEAD 的第 1 个元素 (索引 0) 就是累积天数 TIME
                if doubhead is not None and len(doubhead) > 0:
                    step_day_map[step_idx] = float(doubhead[0])
                else:
                    step_day_map[step_idx] = float(step_idx)
            except Exception as e:
                print(f"警告: 无法读取时间步 {step_idx} 的 DOUBHEAD: {e}")
                
        return step_day_map

    def get_dynamic_datetimes(self) -> Dict[int, datetime]:
        """
        从 Restart 文件的 INTEHEAD 中提取绝对日历时间 (Datetime)。
        非常安全，完全不依赖于外部文件的 DATES 关键字！
        
        Returns:
            Dict[int, datetime]: 格式为 {step_idx: datetime_obj}
        """
        step_date_map = {}
        
        for step_idx, file_path in self.dynamic_files.items():
            try:
                ecl_file = EclFile(str(file_path))
                intehead = self._ecl_get(ecl_file, "INTEHEAD")
                
                # Eclipse 标准：INTEHEAD 的 65, 66, 67 元素分别是 日, 月, 年 (1-based)
                # 对应 Python 中的 0-based 索引为 64, 65, 66
                if intehead is not None and len(intehead) >= 67:
                    day = int(intehead[64])
                    month = int(intehead[65])
                    year = int(intehead[66])
                    
                    # 容错：有些刚初始化的 0 步可能会有不合法的日期如 (0, 0, 0)
                    if year > 0 and month > 0 and day > 0:
                        step_date_map[step_idx] = datetime(year, month, day)
                    else:
                        step_date_map[step_idx] = None
                else:
                    step_date_map[step_idx] = None
            except Exception as e:
                print(f"警告: 无法读取时间步 {step_idx} 的 INTEHEAD 日期: {e}")
                
        return step_date_map

    def get_dynamic_field_names(self, time_step: int) -> List[str]:
        """
        获取特定时间步下，尺寸与网格匹配的有效动态变量名称。
        
        Args:
            time_step: 时间步索引 (如 1)
        
        Returns:
            List[str]: 有效动态场名称列表 (如 ['PRESSURE', 'SWAT'])
        """
        time_step = self._resolve_step(time_step) # 🌟 拦截解析

        dynamic_files = self.dynamic_files
        if time_step not in self.dynamic_files:
            raise ValueError(f"Time step {time_step} not found.")
            
        path = dynamic_files[time_step]
        erst = ERst(str(path))
        step_idx = erst.report_steps[0]
        
        target_sizes = {self.base_active_cells, self.base_actnum.size}
        valid_keys = [
            name for name, arrType, arrSize in erst.arrays(step_idx)
            if arrSize in target_sizes
        ]
        return valid_keys

    def get_dynamic_fields(self, idx_step: int, field_names: List[str], certain_actnum: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取特定时间步的动态场数值，并映射到当前 active 掩码体系。
        
        Args:
            idx_step: 时间步索引
            field_names: 字段关键字列表 (如 ['PRESSURE', 'SWAT'])
            certain_actnum: 需要筛选的网格掩码。如果不传，则返回与 get_model_actnum() 对应的有效数据。
            
        Returns:
            np.ndarray: shape 为 (N_active,) 的 1 维浮点数组，数值完全对齐
        """
        idx_step = self._resolve_step(idx_step)   # 🌟 拦截解析

        dynamic_files = self.dynamic_files
        if idx_step not in dynamic_files:
            raise ValueError(f"Time step {idx_step} not found.")
            
        erst = ERst(str(dynamic_files[idx_step]))
        step_idx = erst.report_steps[0]


        if isinstance(field_names, str):
            field_raw = erst[field_names, step_idx]
            return self._format_field_to_active(field_raw, certain_actnum)
        
        res = []
        for field_name in field_names:
            field_raw = erst[field_name, step_idx]
            res.append(self._format_field_to_active(field_raw, certain_actnum))

        return np.stack(res, axis=0)
    

    def get_static_field_names(self) -> List[str]:
        """
        获取静态模型 (.INIT) 中尺寸与网格匹配的有效静态变量名称。
        
        Returns:
            List[str]: 有效静态场名称列表 (如 ['PORO', 'PERMX'])
        """
        target_sizes = {self.base_active_cells, self.base_actnum.size}
        valid_keys = []
        for k in self.emod.get_list_of_arrays():
            name = k[0]
            try:
                arr = self.emod.get(name)
                if np.size(arr) in target_sizes:
                    valid_keys.append(name)
            except Exception:
                pass
        return valid_keys

    def get_static_fields(self, field_names: List[str], certain_actnum: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取静态模型场数值，并映射到当前 active 掩码体系。
        
        Args:
            field_names: 字段关键字列表 (如 ['PORO', 'PERMX'])
            certain_actnum: 需要筛选的网格掩码。如果不传，则返回全模型的有效数据。
            
        Returns:
            np.ndarray: shape 为 (N_active,) 的 1 维浮点数组
        """
        if isinstance(field_names, str):
            field_raw = self.emod.get(field_names)
            return self._format_field_to_active(field_raw, certain_actnum)
        res = []
        for field_name in field_names:
            field_raw = self.emod.get(field_name)
            res.append(self._format_field_to_active(field_raw, certain_actnum))
        return np.stack(res, axis=0)

    def _format_field_to_active(self, field_raw: np.ndarray, certain_actnum: Optional[np.ndarray]) -> np.ndarray:
        """
        内部工具：强制将 Eclipse 的 Big-Endian 数组清洗为标准 C-Contiguous Float64，
        并精确提取到 specified actnum 的一维长度上。
        """
        actnum = self.base_actnum if certain_actnum is None else certain_actnum
        field_flat = np.array(field_raw, dtype=float, copy=True).flatten()
        
        if field_flat.size == self.base_active_cells:
            # [Optimized] 直接提取，不构建 3D 临时数组
            if certain_actnum is None:
                return field_flat
            
            target_size = np.count_nonzero(actnum)
            return map_sub_active_field_numba(
                self.base_actnum.ravel(), 
                actnum.ravel(), 
                field_flat, 
                target_size
            )
            
        elif field_flat.size == self.base_actnum.size:
            return field_flat[actnum.flatten()]
        else:
            raise ValueError(f"Data size ({field_flat.size}) mismatch with grid dimensions.")

    def get_pv_static_mesh(self, field_names: str|List[str], certain_actnum: Optional[np.ndarray] = None):
        """生成静态属性的 PyVista PolyData 网格对象"""
        quads, quad_cells = self.get_surface_quads(certain_actnum)
        field_1d = self.get_static_fields(field_names, certain_actnum)
        vals_on_faces = field_1d[quad_cells]
        return self._build_pv_mesh(quads, vals_on_faces, field_names)

    def get_pv_dynamic_mesh(self, time_step: int, field_names: str|List[str], certain_actnum: Optional[np.ndarray] = None):
        """生成动态属性的 PyVista PolyData 网格对象"""
        time_step = self._resolve_step(time_step) # 🌟 拦截解析

        quads, quad_cells = self.get_surface_quads(certain_actnum)
        field_1d = self.get_dynamic_fields(time_step, field_names, certain_actnum)
        vals_on_faces = field_1d[quad_cells]
        return self._build_pv_mesh(quads, vals_on_faces, field_names)

    def _build_pv_mesh(self, quads: np.ndarray, values: np.ndarray, field_name: str):
        if quads.size == 0:
            return pv.PolyData()

        origin = np.array(self.get_origin())
        quads_centered = quads - origin
        verts = np.ascontiguousarray(quads_centered.reshape(-1, 3), dtype=np.float64)

        # [Optimized] 直接生成 1D faces
        n_faces = len(quads)
        faces_1d = build_pv_faces_numba(n_faces)

        mesh = pv.PolyData(verts, faces_1d)

        # [Optimized] 原地清理异常值
        safe_values = clean_nans_numba(values)
        mesh.cell_data[field_name] = safe_values
        mesh.set_active_scalars(field_name)

        return mesh

# =============================================================================
# 测试入口
# =============================================================================
if __name__ == "__main__":

    PROJECT_DIR = r"../"
    PROJECT_NAME = "BYEPD93"

    # 初始化数据类
    model = EclipseModelData(PROJECT_DIR, PROJECT_NAME)
    
    print("Origin:", model.get_origin())
    print("Frame Size (Z, Y, X):", model.get_3Dframe_size())
    print("Grid Dim (K, J, I):", model.get_3Dframe_dim())
    # print("day mapping:", model.get_dynamic_days())
    # print("datetime mapping:", model.get_dynamic_datetimes())
    
    # 逻辑处理：做局部裁剪
    actnum = model.get_model_actnum()
    nz, ny, nx = model.get_3Dframe_dim()
    actnum[nz//2:, :, :] = False  # 裁剪掉下半部分

    # 1. 核心解耦：只获取数据对象，不直接渲染
    mesh = model.get_pv_static_mesh("PORO", certain_actnum=actnum)

    # 2. UI 渲染层 (这里模拟桌面端单纯的 Plotter 视窗)
    plotter = pv.Plotter()
    # add_mesh 时可以直接指定颜色映射和是否显示网格线
    plotter.add_mesh(mesh, cmap='jet', show_edges=False, scalar_bar_args={'title': "PORO - Top Half"})
    
    plotter.set_background('lightgray')
    plotter.show()