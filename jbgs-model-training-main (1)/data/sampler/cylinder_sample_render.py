# =============================================================================
# 外部测试渲染器 (MP4 雷达阵列排版 + 3D 物理色散排版)
# =============================================================================
import math
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from data.sampler.cylinder_sample import EclipseDataSampler

def render_polar_mp4(samples: list, max_cols: int = 4, max_rows: int = 3, filename: str = "polar_array.mp4"):
    """雷达阵列排版引擎：支持分页滚动播放，计算全局 Colorbar 极值"""
    if not samples: return
    n_samples = len(samples)
    batch_size = max_cols * max_rows
    num_batches = math.ceil(n_samples / batch_size)
    
    model_nz = samples[0]["model_nz"]
    n_theta = samples[0]["n_theta"]
    r_edges = samples[0]["r_edges"]
    theta_edges = np.linspace(0, 2 * math.pi, n_theta + 1)
    T, R = np.meshgrid(theta_edges, r_edges)
    
    # 1. 🌟 计算全局 Colorbar 极值，确保所有批次的颜色具有绝对可比性
    global_min, global_max = np.inf, -np.inf
    plot_datas = []
    for sample in samples:
        field_t = sample["field_data"].numpy()
        mask_t = sample["mask"].numpy()
        f_data = field_t[0] if field_t.ndim == 4 else field_t
        m_data = mask_t[0] if mask_t.ndim == 4 else mask_t
        
        plot_data = np.where(m_data, f_data, np.nan)
        plot_datas.append(plot_data)
        
        valid_vals = plot_data[~np.isnan(plot_data)]
        if len(valid_vals) > 0:
            global_min = min(global_min, valid_vals.min())
            global_max = max(global_max, valid_vals.max())

    if np.isinf(global_min): global_min, global_max = 0, 1

    # 2. 画布初始化
    actual_cols = min(n_samples, max_cols)
    actual_rows = min(math.ceil(n_samples / actual_cols), max_rows)
    fig, axes = plt.subplots(actual_rows, actual_cols, subplot_kw={'projection': 'polar'}, figsize=(actual_cols * 4, actual_rows * 4))
    if actual_rows * actual_cols == 1: axes = np.array([axes])
    axes = axes.flatten()
    
    caxes = []
    empty_data = np.full((n_theta, len(r_edges)-1), np.nan) # 空白页数据

    for idx, ax in enumerate(axes):
        ax.grid(False)
        ax.spines['polar'].set_visible(False)
        cax = ax.pcolormesh(T, R, empty_data.T, cmap='jet', edgecolors='k', linewidth=0.5, vmin=global_min, vmax=global_max)
        caxes.append(cax)
        # 只生成一个总的 Colorbar
        if idx == len(axes) - 1:
            fig.colorbar(cax, ax=axes, fraction=0.046, pad=0.04, label=samples[0].get("field_name", "Field"))
            
    # 3. 🌟 分页连播更新逻辑
    def update(frame):
        batch_idx = frame // model_nz       # 当前是第几批次
        z_layer = frame % model_nz          # 当前批次的 Z 轴深度
        
        for ax_idx in range(len(axes)):
            global_idx = batch_idx * batch_size + ax_idx
            if global_idx < n_samples:
                sample = samples[global_idx]
                time_info = sample.get("time_info", "Static")
                axes[ax_idx].set_title(f"[{batch_idx+1}/{num_batches}] Well: {sample['well_name']}\n{time_info} | Z: {z_layer + 1}/{model_nz}")
                caxes[ax_idx].set_array(plot_datas[global_idx][z_layer].T.ravel())
            else:
                axes[ax_idx].set_title("")
                caxes[ax_idx].set_array(empty_data.T.ravel())
        return caxes
        
    total_frames = num_batches * model_nz
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=600, blit=False)
    ani.save(filename, writer='ffmpeg')
    print(f"✅ 极坐标 MP4 已成功保存至: {filename} (分 {num_batches} 批次连播)")
    plt.close()


def render_3d_cylinders(samples: list, dealer, dispersion_scale_x: float = 3000.0, dispersion_scale_y: float = 3000.0):
    """3D 高颜值排版：按需延迟构建 (Lazy Build)，极致性能，统一全局 Colorbar"""
    if not samples or "_visual_req" not in samples[0]:
        print("未开启 visualize=True，无渲染资产。")
        return

    model = dealer.model
    well_data = dealer.well_data
    origin_arr = np.array(model.get_origin())
    field_name = samples[0].get("field_name", "Field")

    print(f"正在按需构建 {len(samples)} 个圆柱的 3D 几何资产 (Lazy Evaluation)...")

    # 1. 🌟 按需构建！只有传入的 sample 才会被构建网格，彻底解救算力
    global_min, global_max = np.inf, -np.inf
    for sample in samples:
        req = sample["_visual_req"]
        
        # 还原 3D 掩码
        certain_actnum_2d = np.zeros(model.ny * model.nx, dtype=bool)
        certain_actnum_2d[req["valid_2d_indices"]] = True
        certain_actnum_3d = certain_actnum_2d.reshape(model.ny, model.nx)[np.newaxis, :, :] & model.get_model_actnum()
        
        step_idx = sample.get("step_idx", -1)
        
        # 调用底层生成真正的 PyVista 对象
        if sample.get("time_info") == "Static":
            mesh = model.get_pv_static_mesh(field_name, certain_actnum_3d)
        else:
            mesh = model.get_pv_dynamic_mesh(step_idx, field_name, certain_actnum_3d)
            
        pv_render_items, pv_label_items = well_data.get_pv_well_tracks(
            model=model, 
            step_idx=step_idx, 
            well_names=req["hd_well_names"],
            display_radius=5.0,        
            extend_track_length=200.0, 
            show_perforation=True,     
            show_labels=True, 
            label_scale=14
        )
        
        # 暂存资产到 sample 中供第二步渲染
        sample["_pv_mesh"] = mesh
        sample["_pv_render_items"] = pv_render_items
        sample["_pv_label_items"] = pv_label_items
        
        # 获取 Colorbar 极值
        if field_name in mesh.cell_data:
            vals = mesh.cell_data[field_name]
            if len(vals) > 0:
                global_min = min(global_min, np.nanmin(vals))
                global_max = max(global_max, np.nanmax(vals))

    if np.isinf(global_min): global_min, global_max = 0, 1

    print("正在拉起 PyVista 3D 交互视窗...")
    plotter = pv.Plotter()
    
    n_samples = len(samples)
    cols = math.ceil(math.sqrt(n_samples))
    
    for idx, sample in enumerate(samples):
        col, row = idx % cols, idx // cols
        shift = np.array([col * dispersion_scale_x, row * dispersion_scale_y, 0.0])
        
        # 渲染油藏圆柱
        mesh = sample["_pv_mesh"]
        if mesh.n_points > 0:
            mesh.points = mesh.points + shift 
            plotter.add_mesh(mesh, cmap='jet', show_edges=True, opacity=0.9,
                             scalars=field_name, clim=[global_min, global_max],
                             scalar_bar_args={'title': field_name} if idx == 0 else None)
            
        # 渲染井轨迹
        for item in sample["_pv_render_items"]:
            track_mesh = item["mesh"]
            track_mesh.points = track_mesh.points - origin_arr + shift
            plotter.add_mesh(track_mesh, **item["kwargs"])
            
        # 渲染标签
        for lbl in sample["_pv_label_items"]:
            lbl_pts = lbl["points"] - origin_arr + shift
            texts = lbl["texts"]
            for i, w_name in enumerate(texts):
                if w_name == sample["well_name"]:
                    # label_text = f"{w_name}\n{sample.get('time_info', '')}"
                    label_text = w_name
                    plotter.add_point_labels(lbl_pts[i:i+1], [label_text], text_color=lbl["kwargs"]["text_color"], font_size=16, shape_opacity=0.4)
                else:
                    plotter.add_point_labels(lbl_pts[i:i+1], [w_name], text_color=lbl["kwargs"]["text_color"], font_size=10, shape_opacity=0.1)

    plotter.set_background('white')
    plotter.add_axes()
    plotter.show()


if __name__ == "__main__":
    PROJECT_DIR, PROJECT_NAME = r"D:\\YihengZhu\\jbgs-lma\\proj-phase1\\eclipse-models\\1c\\water_flooding", "BYEPD93"
    
    print("="*60)
    print("🚀 启动流式极坐标发牌人与可视化升华测试...")
    
    dealer = EclipseDataSampler(PROJECT_DIR, PROJECT_NAME, seed=42)
    
    # 获取一部分活井用于测试阵列展示
    test_wells = set(np.random.choice(dealer.split_wells['all'], size=6, replace=False)) # 抽取前 6 口井进行排版
    print(f"\n锁定测试目标井: {test_wells}")

    # ==========================================
    # 测试 1：动态场，在特定时间步上的快照阵列
    # ==========================================
    target_step = 50
    print(f"\n⏳ 正在抽取第 {target_step} 步的动态 SWAT 和 PRESSURE 场...")
    dynamic_gen = dealer.iter_dynamic(
        field_names=["SWAT", "PRESSURE"], 
        mode="all", 
        visualize=True, 
        shuffle=False, 
        target_steps=[target_step] # 🌟 精确制导
    )
    
    samples_dynamic = []
    for sample in dynamic_gen:
        if sample["well_name"] in test_wells:
            samples_dynamic.append(sample)
            
    print(f"成功收集 {len(samples_dynamic)} 个动态圆柱样本。")
    if samples_dynamic:
        # 生成雷达阵列排版 MP4 (最大每行 3 个)
        render_polar_mp4(samples_dynamic, max_cols=3, filename=f"dynamic_step{target_step}_radar.mp4")
        
        # 3D 物理色散渲染，X/Y 方向各自拉开 3000m 的距离防重叠
        render_3d_cylinders(samples_dynamic, dealer.model.get_origin(), dispersion_scale_x=3000.0, dispersion_scale_y=3000.0)

    # ==========================================
    # 测试 2：静态场的快照阵列
    # ==========================================
    print(f"\n⏳ 正在抽取静态 PORO 场...")
    static_gen = dealer.iter_static(
        field_name="PORO", 
        mode="all", 
        visualize=True, 
        # visualize=False, 
        shuffle=False
    )
    
    samples_static = []
    for sample in static_gen:
        if sample["well_name"] in test_wells:
            samples_static.append(sample)
        
    print(f"成功收集 {len(samples_static)} 个静态圆柱样本。")

    if samples_static:
        print("\n 开始渲染静态 PORO 场的极坐标阵列 MP4...")
        render_polar_mp4(samples_static, max_rows=8, max_cols=8, filename="static_poro_radar.mp4")
        print("\n 开始渲染静态 PORO 场的 3D 色散排版...")
        render_3d_cylinders(samples_static, dealer, dispersion_scale_x=3000.0, dispersion_scale_y=3000.0)