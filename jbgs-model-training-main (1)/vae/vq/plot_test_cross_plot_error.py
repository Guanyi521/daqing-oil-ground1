import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation

def common_colorbar(fig, time_vector, cmap, norm, shift_x=0.0, shift_y=-0.03):
    """
    绘制水平连续 Colorbar，表示发牌的时间步或物理时间 (Days)
    """
    base_left, base_bottom, base_width, base_height = 0.15, 0.05, 0.7, 0.03
    cax_pos = [base_left + shift_x, base_bottom + shift_y, base_width, base_height]
    cax = fig.add_axes(cax_pos)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_label("Simulation Time Step / Days", fontsize=16, labelpad=10)
    cax.tick_params(labelsize=14)
    return cax

def _draw_spatial_crossplot_grid(fig, axes, true_vals, pred_vals, mask_vals, 
                                 time_vector, theta_indices, r_indices, z_layer_id, 
                                 field_name, cmap, norm):
    """
    内部核心引擎：在指定的 Figure 和 Axes 上绘制 m_rows * n_cols 的阵列。
    避免代码在 png 和 mp4 函数中重复。
    """
    m_rows = len(r_indices)
    n_cols = len(theta_indices)
    
    # 将时间映射为颜色数组 (Vectorized 提速，抛弃低效的 for 循环画点)
    colors = cmap(norm(time_vector))

    for r_idx in range(m_rows):
        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]
            ax.clear() # 为 MP4 刷新做准备
            
            R_pos = r_indices[r_idx]
            Theta_pos = theta_indices[c_idx]
            
            # 取出当前 (Z, Theta, R) 空间点的所有 n_test 张牌的数据
            # shape 均为 (n_test,)
            x_true = true_vals[:, z_layer_id, Theta_pos, R_pos]
            y_pred = pred_vals[:, z_layer_id, Theta_pos, R_pos]
            valid_mask = mask_vals[:, z_layer_id, Theta_pos, R_pos] > 0.5
            
            # 过滤掉死网格或被 mask 掉的数据
            x_val = x_true[valid_mask]
            y_val = y_pred[valid_mask]
            c_val = colors[valid_mask]
            
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            
            if len(x_val) > 0:
                ax.scatter(
                    x_val, y_val,
                    facecolors='none', 
                    edgecolors=c_val,
                    s=40, alpha=0.8, linewidths=1.2
                )
                
                # 画 y=x 的参考线
                min_v = min(np.nanmin(x_val), np.nanmin(y_val))
                max_v = max(np.nanmax(x_val), np.nanmax(y_val))
                
                # 留出一点边距
                margin = (max_v - min_v) * 0.05 if max_v > min_v else 0.1
                ax.plot([min_v - margin, max_v + margin], 
                        [min_v - margin, max_v + margin], 'k--', lw=1.2, zorder=10)
            
            # 设置子图标题与刻度
            ax.set_title(f"θ={Theta_pos}, R={R_pos}", fontsize=12)
            ax.grid(alpha=0.45, zorder=-10, linestyle='--', linewidth=1.0)
            ax.tick_params(labelsize=10)

    # 全局坐标轴标签
    for r in range(m_rows):
        axes[r, 0].set_ylabel(f"Pred {field_name}", fontsize=14)
    for c in range(n_cols):
        axes[-1, c].set_xlabel(f"True {field_name}", fontsize=14)

def plot_spatial_error_matrix_png(
        true_vals: np.ndarray, pred_vals: np.ndarray, mask_vals: np.ndarray,
        time_vector: np.ndarray, theta_indices: list, r_indices: list,
        z_layer_id: int, field_name: str, 
        color_map='rainbow', dpi: int=300, save_path='crossplot_spatial.png'):
    """
    生成特定 Z 层的 m_rows * n_cols 误差阵列图 (PNG)
    
    参数:
        true_vals, pred_vals: shape (n_test, nz, n_theta, n_R)
        mask_vals: shape (n_test, nz, n_theta, n_R) 有效网格掩码
        time_vector: shape (n_test,) 对应每次采样的时间步或天数
        theta_indices: 列索引列表，如 [0, 3, 6, 9]
        r_indices: 行索引列表，如 [0, 3, 6]
    """
    m_rows, n_cols = len(r_indices), len(theta_indices)
    fig_w, fig_h = n_cols * 2.8, m_rows * 2.8

    cmap = plt.get_cmap(color_map)
    norm = Normalize(vmin=np.min(time_vector), vmax=np.max(time_vector))

    fig, axes = plt.subplots(m_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, dpi=dpi)

    _draw_spatial_crossplot_grid(
        fig, axes, true_vals, pred_vals, mask_vals, time_vector, 
        theta_indices, r_indices, z_layer_id, field_name, cmap, norm
    )

    plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.15, wspace=0.25, hspace=0.35)
    
    fig.suptitle(f"{field_name} Error Cross-Plot | Z Layer: {z_layer_id}", fontsize=20, fontweight='bold')
    common_colorbar(fig, time_vector, cmap, norm)

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Spatial cross-plot saved to: {save_path}")

def animate_spatial_error_matrix_mp4(
        true_vals: np.ndarray, pred_vals: np.ndarray, mask_vals: np.ndarray,
        time_vector: np.ndarray, theta_indices: list, r_indices: list,
        z_layer_ids: list, field_name: str, 
        color_map='rainbow', dpi: int=200, save_path='crossplot_spatial.mp4'):
    """
    生成遍历多个 Z 层的 m_rows * n_cols 误差阵列视频 (MP4)
    """
    m_rows, n_cols = len(r_indices), len(theta_indices)
    fig_w, fig_h = n_cols * 2.8, m_rows * 2.8

    cmap = plt.get_cmap(color_map)
    norm = Normalize(vmin=np.min(time_vector), vmax=np.max(time_vector))

    fig, axes = plt.subplots(m_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, dpi=dpi)
    plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.15, wspace=0.25, hspace=0.35)
    
    common_colorbar(fig, time_vector, cmap, norm)
    title_text = fig.suptitle("", fontsize=20, fontweight='bold')

    def update(frame):
        z_idx = z_layer_ids[frame]
        title_text.set_text(f"{field_name} Error Cross-Plot | Z Layer: {z_idx}")
        _draw_spatial_crossplot_grid(
            fig, axes, true_vals, pred_vals, mask_vals, time_vector, 
            theta_indices, r_indices, z_idx, field_name, cmap, norm
        )
        return axes.flatten().tolist() + [title_text]

    print(f"🎥 开始渲染 MP4 视频，共 {len(z_layer_ids)} 帧...")
    ani = animation.FuncAnimation(fig, update, frames=len(z_layer_ids), interval=800, blit=False)
    ani.save(save_path, writer='ffmpeg')
    plt.close(fig)
    print(f"✅ Spatial cross-plot MP4 saved to: {save_path}")


if __name__ == "__main__":
    import time
    import torch
    import yaml
    from eclipse.utils import EclipseDataSampler
    
    # 🚨 请根据您的实际相对/绝对路径修改导入，这是之前写好的模型类
    from vae.vq.vq_vae import VQVAE3D  

    # 1. 设定项目路径与模型路径
    PROJECT_DIR = r"E:/tocug/1c/water_196311-199307"
    PROJECT_NAME = "BYEPD93"
    CONFIG_PATH = "vae/vq/vq_vae_config.yaml"
    CKPT_PATH = "checkpoints/2026-03-16_14.24.47_pressure_swat_best.ckpt"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("🚀 启动空间误差 Cross-plot 阵列绘图 (真实模型推理版)...")
    
    # 2. 🌟 加载模型配置与权重 (加入 strict=False 绕过 grid_3d 报警)
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    print(f"📦 正在加载模型权重: {CKPT_PATH}")
    model = VQVAE3D.load_from_checkpoint(CKPT_PATH, strict=False)
    model.eval()
    model.to(device)

    # 3. 初始化发牌人
    dealer = EclipseDataSampler(PROJECT_DIR, PROJECT_NAME, seed=42)
    target_fields = ["PRESSURE", "SWAT"]  # 🌟 明确请求两个通道
    
    test_steps = dealer.model.get_dynamic_steps()[::5][:10]
    
    dynamic_gen = dealer.iter_dynamic(
        field_names=target_fields, 
        mode="val", 
        visualize=False, 
        shuffle=False,
        target_steps=test_steps
    )
    
    true_list, pred_list, mask_list, time_list = [], [], [], []
    
    print(f"⏳ 正在从发牌人抽取数据并执行 VQ-VAE 推理...")
    t0 = time.time()
    with torch.no_grad():
        for sample in dynamic_gen:
            # 🌟 强制 .float() 防止 double 与 float 不匹配报错
            # x_true_t shape: (1, 2, nz, n_theta, n_R)
            x_true_t = sample["field_data"].unsqueeze(0).to(device).float() 
            # mask_t shape: (1, 1, nz, n_theta, n_R)
            mask_t = sample["mask"].unsqueeze(0).to(device).float()       
            
            # 🌟 核心：执行真实模型前向传播，得到双通道重构场
            x_pred_t, _, _, _, _ = model(x_true_t, mask_t)
            
            # 剥离掉最外层的 Batch 维度
            true_list.append(x_true_t.cpu().numpy()[0])   # (2, nz, n_theta, n_R)
            pred_list.append(x_pred_t.cpu().numpy()[0])   # (2, nz, n_theta, n_R)
            mask_list.append(mask_t.cpu().numpy()[0])     # (1, nz, n_theta, n_R)
            
            # 获取物理天数
            step_idx = sample["step_idx"]
            days = dealer.model.get_dynamic_days().get(step_idx, float(step_idx))
            time_list.append(days)

    # 4. 堆叠大矩阵
    # true_vals_all shape = (n_test, 2, nz, n_theta, n_R)
    true_vals_all = np.stack(true_list, axis=0)
    pred_vals_all = np.stack(pred_list, axis=0)
    mask_vals_all = np.stack(mask_list, axis=0) 
    time_vector = np.array(time_list)
    
    print(f"✅ 推理准备完毕，耗时 {time.time()-t0:.2f} 秒. 总体 Tensor 形状: {true_vals_all.shape}")
    
    theta_indices = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    r_indices = [0, 3, 6, 9]
    nz_total = true_vals_all.shape[2]
    test_z = nz_total // 2
    
    # =========================================================
    # 🌟 5. 分离通道，通过循环依次绘制 Pressure 和 SWAT 的图像
    # =========================================================
    for c_idx, field_name in enumerate(target_fields):
        print(f"\n" + "="*40)
        print(f"🎯 开始处理通道 [{c_idx}]: {field_name}")
        
        # 切片提取单通道数据 -> shape (n_test, nz, n_theta, n_R)
        c_true = true_vals_all[:, c_idx, ...]
        c_pred = pred_vals_all[:, c_idx, ...]
        c_mask = mask_vals_all[:, 0, ...]  # 掩码大家共用第 0 个通道
        
        # A. 生成特定 Z 层的静态图
        png_path = f"{field_name.lower()}_error_z{test_z}.png"
        print(f"🎨 渲染静态图 (Z={test_z}) -> {png_path}")
        plot_spatial_error_matrix_png(
            c_true, c_pred, c_mask, time_vector, 
            theta_indices, r_indices, z_layer_id=test_z, 
            field_name=field_name, save_path=png_path
        )
        
        # B. 生成遍历所有 Z 层的动态视频 (MP4)
        mp4_path = f"{field_name.lower()}_error_scan.mp4"
        print(f"🎬 渲染动态视频扫描 -> {mp4_path}")
        animate_spatial_error_matrix_mp4(
            c_true, c_pred, c_mask, time_vector, 
            theta_indices, r_indices, z_layer_ids=list(range(nz_total)), 
            field_name=field_name, save_path=mp4_path
        )

    print("\n🎉 双通道交叉误差绘图任务全部完成！")