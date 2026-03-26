import math
from typing import Tuple
import numpy as np
from numba import njit, prange

# =============================================================================
# Numba 极速外表面提取核 (独立于类外部，以获取最佳编译性能)
# =============================================================================

@njit
def build_act_remap(actnum: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    """构建 (nz, ny, nx) 到一维 active 索引的映射矩阵"""
    act_remap = np.full((nz, ny, nx), -1, dtype=np.int32)
    idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if actnum[k, j, i]:
                    act_remap[k, j, i] = idx
                    idx += 1
    return act_remap

@njit
def compute_face_offsets(actnum: np.ndarray, nx: int, ny: int, nz: int) -> Tuple[np.ndarray, int]:
    """预计算每个有效网格单元的暴露面个数，并计算全局起始偏移量"""
    offsets = np.zeros((nz, ny, nx), dtype=np.int32)
    total = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if actnum[k, j, i]:
                    c = 0
                    if i == 0 or not actnum[k, j, i - 1]: c += 1
                    if i == nx - 1 or not actnum[k, j, i + 1]: c += 1
                    if j == 0 or not actnum[k, j - 1, i]: c += 1
                    if j == ny - 1 or not actnum[k, j + 1, i]: c += 1
                    if k == 0 or not actnum[k - 1, j, i]: c += 1
                    if k == nz - 1 or not actnum[k + 1, j, i]: c += 1
                    offsets[k, j, i] = total
                    total += c
    return offsets, total

@njit(parallel=True, fastmath=True)
def get_boundary_quads_numba(
    coord_1d: np.ndarray, zcorn_1d: np.ndarray, actnum: np.ndarray, 
    act_remap: np.ndarray, nx: int, ny: int, nz: int, 
    nblocks: int, radial: bool, eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """直接从 COORD / ZCORN 提取暴露在空气中的四边形顶点，并返回所属细胞索引"""
    offsets, total_faces = compute_face_offsets(actnum, nx, ny, nz)
    quads = np.empty((total_faces, 4, 3), dtype=np.float64)
    quad_cells = np.empty(total_faces, dtype=np.int32)

    pillars_per_block = (nx + 1) * (ny + 1)
    block_size = pillars_per_block * 6
    nx1_6 = (nx + 1) * 6
    ncells = nz * ny * nx

    f_idx = np.array([
        [0, 2, 6, 4], [1, 5, 7, 3], [0, 4, 5, 1],
        [2, 3, 7, 6], [0, 1, 3, 2], [4, 6, 7, 5],
    ], dtype=np.int32)

    for idx in prange(ncells):
        i, rem = idx % nx, idx // nx
        j, k = rem % ny, rem // ny

        if not actnum[k, j, i]: continue

        c_idx = offsets[k, j, i]
        act_id = act_remap[k, j, i]

        res_shift = k * block_size if nblocks == nz else 0
        p0 = res_shift + j * nx1_6 + i * 6
        pinds = np.array([p0, p0 + 6, p0 + nx1_6, p0 + nx1_6 + 6], dtype=np.int64)

        z0 = k * (nx * ny * 8) + j * (nx * 4) + i * 2
        base_top = z0 + nx * ny * 4
        zinds = np.array([z0, z0+1, z0+nx*2, z0+nx*2+1, base_top, base_top+1, base_top+nx*2, base_top+nx*2+1], dtype=np.int64)

        Z = np.empty(8, dtype=np.float64)
        for n in range(8): Z[n] = zcorn_1d[zinds[n]]

        X, Y = np.empty(8, dtype=np.float64), np.empty(8, dtype=np.float64)
        for n in range(4):
            p = pinds[n]
            xt, yt, zt = coord_1d[p], coord_1d[p+1], coord_1d[p+2]
            xb, yb, zb = coord_1d[p+3], coord_1d[p+4], coord_1d[p+5]

            if radial:
                th_t, th_b = yt * (math.pi / 180.0), yb * (math.pi / 180.0)
                xt, yt = xt * math.cos(th_t), xt * math.sin(th_t)
                xb, yb = xb * math.cos(th_b), xb * math.sin(th_b)

            denom = zt - zb
            if abs(denom) <= eps:
                X[n], Y[n], X[n+4], Y[n+4] = xt, yt, xt, yt
            else:
                r0, r1 = (zt - Z[n]) / denom, (zt - Z[n+4]) / denom
                X[n], Y[n] = xt + (xb - xt) * r0, yt + (yb - yt) * r0
                X[n+4], Y[n+4] = xt + (xb - xt) * r1, yt + (yb - yt) * r1

        C8 = np.empty((8, 3), dtype=np.float64)
        for n in range(8):
            C8[n, 0], C8[n, 1], C8[n, 2] = X[n], Y[n], Z[n]

        faces_cond = [
            i == 0 or not actnum[k, j, i - 1], i == nx - 1 or not actnum[k, j, i + 1],
            j == 0 or not actnum[k, j - 1, i], j == ny - 1 or not actnum[k, j + 1, i],
            k == 0 or not actnum[k - 1, j, i], k == nz - 1 or not actnum[k + 1, j, i]
        ]

        for f, cond in enumerate(faces_cond):
            if cond:
                for m in range(4):
                    for d in range(3):
                        quads[c_idx, m, d] = C8[f_idx[f, m], d]
                quad_cells[c_idx] = act_id
                c_idx += 1

    return quads, quad_cells


@njit(parallel=True)
def build_pv_faces_numba(n_faces: int) -> np.ndarray:
    """直接在 1D 内存中并行构建 PyVista 的多边形索引数组"""
    faces_1d = np.empty(n_faces * 5, dtype=np.int64)
    for i in prange(n_faces):
        base_idx = i * 5
        vert_idx = i * 4
        faces_1d[base_idx] = 4
        faces_1d[base_idx + 1] = vert_idx
        faces_1d[base_idx + 2] = vert_idx + 1
        faces_1d[base_idx + 3] = vert_idx + 2
        faces_1d[base_idx + 4] = vert_idx + 3
    return faces_1d

@njit(parallel=True)
def clean_nans_numba(values: np.ndarray) -> np.ndarray:
    """并行替换 NaN/Inf 为 0.0，避免 np.nan_to_num 的额外开销"""
    out = np.empty_like(values)
    for i in prange(values.size):
        v = values[i]
        if np.isnan(v) or np.isinf(v):
            out[i] = 0.0
        else:
            out[i] = v
    return out

@njit(fastmath=True)
def compute_bounds_numba(quads: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """单次遍历找出包围盒，避免分配巨大的布尔掩码内存"""
    min_vals = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    max_vals = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    
    n_faces = quads.shape[0]
    for i in range(n_faces):
        for j in range(4):
            for d in range(3):
                val = quads[i, j, d]
                if np.isfinite(val):
                    if val < min_vals[d]: min_vals[d] = val
                    if val > max_vals[d]: max_vals[d] = val
                    
    return min_vals, max_vals


@njit
def map_sub_active_field_numba(base_act_1d: np.ndarray, sub_act_1d: np.ndarray, field_flat: np.ndarray, out_size: int) -> np.ndarray:
    """直接在两个 active 掩码之间映射数据，跳过全尺寸 3D 零矩阵的分配"""
    out = np.empty(out_size, dtype=np.float64)
    base_ptr = 0
    out_ptr = 0
    
    for i in range(base_act_1d.size):
        if base_act_1d[i]:
            val = field_flat[base_ptr]
            base_ptr += 1
            if sub_act_1d[i]:
                out[out_ptr] = val
                out_ptr += 1
                
    return out


@njit(parallel=True, fastmath=True)
def compute_cell_centers_numba(
    I_arr: np.ndarray, J_arr: np.ndarray, K_arr: np.ndarray,
    coord_1d: np.ndarray, zcorn_1d: np.ndarray,
    nx: int, ny: int, nz: int, nblocks: int, radial: bool, eps: float
) -> np.ndarray:
    """直接从 COORD/ZCORN 批量计算多个 (I, J, K) 对应的网格单元中心 (X, Y, Z)"""
    N = len(I_arr)
    centers = np.empty((N, 3), dtype=np.float64)

    pillars_per_block = (nx + 1) * (ny + 1)
    block_size = pillars_per_block * 6
    nx1_6 = (nx + 1) * 6

    for idx in prange(N):
        # Eclipse COMPDAT 的 I, J, K 是从 1 开始的，转换为 0-based 索引
        i = I_arr[idx] - 1
        j = J_arr[idx] - 1
        k = K_arr[idx] - 1

        # 找到对应 pillar 的坐标索引
        res_shift = k * block_size if nblocks == nz else 0
        p0 = res_shift + j * nx1_6 + i * 6
        pinds = np.array([p0, p0 + 6, p0 + nx1_6, p0 + nx1_6 + 6], dtype=np.int64)

        # 找到对应 ZCORN 的索引
        z0 = k * (nx * ny * 8) + j * (nx * 4) + i * 2
        base_top = z0 + nx * ny * 4
        zinds = np.array([z0, z0+1, z0+nx*2, z0+nx*2+1, 
                          base_top, base_top+1, base_top+nx*2, base_top+nx*2+1], dtype=np.int64)

        # 提取 8 个顶点的 Z 坐标
        Z = np.empty(8, dtype=np.float64)
        for n in range(8): 
            Z[n] = zcorn_1d[zinds[n]]

        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0

        # 对 4 根柱子进行插值，计算 8 个顶点的 X, Y
        for n in range(4):
            p = pinds[n]
            xt, yt, zt = coord_1d[p], coord_1d[p+1], coord_1d[p+2]
            xb, yb, zb = coord_1d[p+3], coord_1d[p+4], coord_1d[p+5]

            if radial:
                th_t, th_b = yt * (math.pi / 180.0), yb * (math.pi / 180.0)
                xt, yt = xt * math.cos(th_t), xt * math.sin(th_t)
                xb, yb = xb * math.cos(th_b), xb * math.sin(th_b)

            denom = zt - zb
            
            # 顶部顶点
            z_top = Z[n]
            if abs(denom) <= eps:
                x_top, y_top = xt, yt
            else:
                r0 = (zt - z_top) / denom
                x_top, y_top = xt + (xb - xt) * r0, yt + (yb - yt) * r0

            # 底部顶点
            z_bot = Z[n+4]
            if abs(denom) <= eps:
                x_bot, y_bot = xt, yt
            else:
                r1 = (zt - z_bot) / denom
                x_bot, y_bot = xt + (xb - xt) * r1, yt + (yb - yt) * r1

            # 累加 8 个顶点的坐标
            sum_x += (x_top + x_bot)
            sum_y += (y_top + y_bot)
            sum_z += (z_top + z_bot)

        # 单元中心即为 8 个顶点坐标的平均值
        centers[idx, 0] = sum_x / 8.0
        centers[idx, 1] = sum_y / 8.0
        centers[idx, 2] = sum_z / 8.0

    return centers