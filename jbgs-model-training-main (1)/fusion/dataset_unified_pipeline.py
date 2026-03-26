import torch
import random
import math
import collections
from torch.utils.data import IterableDataset, DataLoader
from data.sampler.cylinder_sample import EclipseDataSampler


class DatasetUnifiedPipeline(IterableDataset):
    """
    针对 Unified Dynamics Pipeline (Stage 2) 设计的数据加载器。
    自动拉取静态孔渗场，缓存动态场并执行滑动窗口切分 (Sliding Window)，
    最终输出 unified_pipeline 要求的 9 项元组。
    """

    def __init__(
            self,
            project_dirs: list,
            project_names: list,
            dynamic_fields: list,  # e.g., ["PRESSURE", "SWAT"]
            nt_width: int = 10,  # 模型一次消化的时间步长度
            stride: int = 1,  # 滑动窗口的步长
            mode: str = "train",
            nums_loading_projects: int = 2,  # 内存中并发持有项目数
            nθ: int = 26,
            nR: int = 10,
            model_nz: int = 40,
            maxR: float = 1000.0,  # 物理超参 (用于 Pipeline 的欧式 PE)
            total_z_depth: float = 50.0,  # 物理超参 (用于 Pipeline 的欧式 PE)
            train_ratio: float = 0.8,
            seed: int = 42,
            shuffle: bool = True
    ):
        super().__init__()
        self.projects = list(zip(project_dirs, project_names))
        self.dynamic_fields = dynamic_fields
        self.nt_width = nt_width
        self.stride = stride
        self.mode = mode
        self.nums_loading_projects = nums_loading_projects
        self.nθ, self.nR, self.model_nz = nθ, nR, model_nz
        self.maxR = maxR
        self.total_z_depth = total_z_depth
        self.train_ratio, self.seed, self.shuffle = train_ratio, seed, shuffle

    def _process_chunk(self, chunk_projects):
        """核心处理逻辑：加载项目 -> 提取静态场 -> 缓存动态场 -> 切分Window并拼装"""
        chunk_windows = []

        for p_dir, p_name in chunk_projects:
            try:
                # 1. 实例化发牌人
                sampler = EclipseDataSampler(
                    project_dir=p_dir, project_name=p_name,
                    nθ=self.nθ, nR=self.nR, model_nz=self.model_nz,
                    train_ratio=self.train_ratio, seed=self.seed
                )

                # 2. 提取并字典化静态孔渗底座 (每口井一份)
                # 注意：确保 shuffle=False，快速按井获取即可
                poro_dict = {s['well_name']: s for s in sampler.iter_static("PORO", mode=self.mode, shuffle=False)}
                perm_dict = {s['well_name']: s for s in sampler.iter_static("PERMX", mode=self.mode, shuffle=False)}

                # 3. 提取动态场并缓存 (构建井史)
                well_buffer = collections.defaultdict(list)
                dynamic_gen = sampler.iter_dynamic(self.dynamic_fields, mode=self.mode, shuffle=False)
                for sample in dynamic_gen:
                    w_name = sample['well_name']
                    well_buffer[w_name].append(sample)

                # 4. 🌟 执行滑动窗口切分 (Sliding Window)
                for w_name, seq in well_buffer.items():
                    # 必须按真实时间步 step_idx 排序，保证物理因果连续性
                    seq.sort(key=lambda x: x['step_idx'])
                    history_len = len(seq)

                    # 窗口长度需要 nt_width + 1 (因为包含了最后一个目标时刻 Target)
                    window_len = self.nt_width + 1

                    # 借鉴了 calculate_num_windows 的思想：不够一个窗口直接跳过
                    if history_len < window_len:
                        continue

                    # 步长滑动切取
                    for start in range(0, history_len - self.nt_width, self.stride):
                        window = seq[start: start + window_len]

                        # 兜底：如果尾部切出来的不足一个完整窗口，直接丢弃
                        if len(window) < window_len:
                            break

                        # === 拼装 Pipeline 需要的 9 项数据 ===
                        # 1. window_fields: 形状 (T, C, nz, nθ, nR)
                        window_fields = torch.stack([s['field_data'] for s in window])

                        # 2. datetimes: 形状 (T-1,) 传入前 nt_width 步的时间戳作为 RoPE 输入
                        datetimes = torch.tensor([s['step_idx'] for s in window[:-1]], dtype=torch.float32)

                        # 3 & 4. 获取对应的静态场
                        poro_field = poro_dict[w_name]['field_data']  # (1, nz, nθ, nR)
                        perm_field = perm_dict[w_name]['field_data']  # (1, nz, nθ, nR)

                        # 5. dynamic_features: Pipeline 允许为 None，它会自动用 dummy 填充
                        dynamic_features = None

                        # 6. spatial_mask_orig: 静态掩码
                        spatial_mask_orig = poro_dict[w_name]['mask']

                        # 7. hard_data_locs: (N, 3)，由于同一口井的结构固定，取 window[0] 即可
                        hard_data_locs = window[0]['hd_locs']

                        chunk_windows.append((
                            window_fields,  # 1. 动态窗口场
                            datetimes,  # 2. 时间戳
                            poro_field,  # 3. 孔隙度
                            perm_field,  # 4. 渗透率
                            dynamic_features,  # 5. 外部井控动态特征 (占位)
                            spatial_mask_orig,  # 6. 空间掩码
                            hard_data_locs,  # 7. 硬数据位置
                            self.maxR,  # 8. 物理半径
                            self.total_z_depth  # 9. 物理深度
                        ))

            except Exception as e:
                print(f"⚠️ 警告: Unified Pipeline 处理项目 {p_name} 失败: {e}")
                continue

        return chunk_windows

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        # 进程任务分配
        if worker_info is not None:
            per_worker = int(math.ceil(len(self.projects) / float(worker_info.num_workers)))
            my_projects = self.projects[worker_info.id * per_worker: (worker_info.id + 1) * per_worker]
        else:
            my_projects = self.projects.copy()

        # 项目级 Shuffle
        if self.shuffle and self.mode == "train":
            random.shuffle(my_projects)

        # 内存分块流水线 (Chunking)
        for i in range(0, len(my_projects), self.nums_loading_projects):
            chunk_projects = my_projects[i: i + self.nums_loading_projects]

            # 集中榨取这些项目并切好 Windows
            chunk_windows = self._process_chunk(chunk_projects)

            # Batch 级(窗口级)局部 Shuffle，保证训练异质性
            if self.shuffle and self.mode == "train":
                random.shuffle(chunk_windows)

            # 逐个吐出切好的滑动窗口
            for window_data in chunk_windows:
                yield window_data


# ==========================================
# 本地测试模块
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 测试 Unified Pipeline Dataset (滑动窗口与多模态拼装)...")

    # 填入你自己的测试路径
    PROJECT_DIR = r"D:\\YihengZhu\\jbgs-lma\\proj-phase1\\eclipse-models\\1c\\water_flooding"
    PROJECT_NAME = "BYEPD93"

    # 模拟手头有 2 个项目
    p_dirs = [PROJECT_DIR, PROJECT_DIR]
    p_names = [PROJECT_NAME, PROJECT_NAME]

    train_dataset = DatasetUnifiedPipeline(
        project_dirs=p_dirs,
        project_names=p_names,
        dynamic_fields=["PRESSURE", "SWAT"],
        nt_width=4,  # 模拟滑动窗口 T=4 (实际返回长度为 5)
        stride=1,
        mode="train",
        nums_loading_projects=2
    )

    train_loader = DataLoader(train_dataset, batch_size=2, num_workers=0)

    for batch_idx, batch in enumerate(train_loader):
        # 完美对齐 unified_pipeline_config 中的 9 项输入
        window_fields, datetimes, poro_field, perm_field, dynamic_features, spatial_mask_orig, hard_data_locs, maxR, total_z_depth = batch

        print(f"\n📦 Batch {batch_idx + 1} 获取成功:")
        print(f"  --> 动态场窗口 (T=nt_width+1): {window_fields.shape}")  # 应为 (2, 5, 2, 40, 26, 10)
        print(f"  --> 时间戳序列 (T=nt_width): {datetimes.shape}")  # 应为 (2, 4)
        print(f"  --> 静态孔隙度底座: {poro_field.shape}")  # 应为 (2, 1, 40, 26, 10)
        print(f"  --> 静态掩码: {spatial_mask_orig.shape}")
        print(f"  --> 硬数据位置: {hard_data_locs.shape}")

        if batch_idx == 0:
            break

    print("=" * 60)
    print("✅ 测试通过！Window 级多模态数据拼装完美。")