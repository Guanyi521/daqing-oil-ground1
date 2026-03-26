import torch
import random
import math
from torch.utils.data import IterableDataset, DataLoader
from eclipse.utils import EclipseDataSampler

class DatasetVQVAE3D(IterableDataset):
    """
    动态属性 VQ-VAE 专用加载器 (P, Sw, Co, Cg, Psi)。
    仅调用 iter_dynamic，包含时间步 Shuffle 与多通道联合输出。
    """
    def __init__(
        self, 
        project_dirs: list,             
        project_names: list,            
        field_names: list,              # 动态场通常是多属性联合，如 ["PRESSURE", "SWAT"]
        mode: str = "train",            
        nums_loading_projects: int = 4, 
        nθ: int = 26, nR: int = 10, model_nz: int = 40, 
        train_ratio: float = 0.8, seed: int = 42, shuffle: bool = True
    ):
        super().__init__()
        self.projects = list(zip(project_dirs, project_names))
        self.field_names = field_names
        self.mode = mode
        self.nums_loading_projects = nums_loading_projects
        self.nθ, self.nR, self.model_nz = nθ, nR, model_nz
        self.train_ratio, self.seed, self.shuffle = train_ratio, seed, shuffle

    def _create_dealer(self, p_dir, p_name):
        try:
            sampler = EclipseDataSampler(
                project_dir=p_dir, project_name=p_name, 
                nθ=self.nθ, nR=self.nR, model_nz=self.model_nz, 
                train_ratio=self.train_ratio, seed=self.seed
            )
            # 🌟 专门调用动态接口
            return sampler.iter_dynamic(self.field_names, mode=self.mode, visualize=False, shuffle=self.shuffle)
        except Exception as e:
            print(f"⚠️ 警告: 加载动态项目 {p_name} 失败: {e}")
            return None

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = int(math.ceil(len(self.projects) / float(worker_info.num_workers)))
            my_projects = self.projects[worker_info.id * per_worker : (worker_info.id + 1) * per_worker]
        else:
            my_projects = self.projects.copy()

        if self.shuffle and self.mode == "train":
            random.shuffle(my_projects)

        pending_projects = my_projects.copy()
        active_dealers = []

        while len(active_dealers) < self.nums_loading_projects and pending_projects:
            p_dir, p_name = pending_projects.pop(0)
            dealer = self._create_dealer(p_dir, p_name)
            if dealer: active_dealers.append(dealer)

        while active_dealers:
            idx = random.randrange(len(active_dealers))
            dealer = active_dealers[idx]
            try:
                sample = next(dealer)
                # 🌟 严格对齐 VQVAE3D.training_step: x, hard_data_locs, mask
                # 注意：如果后续你需要用到 hd_locs_val，可以直接在这里 yield 出四个元素，并同步修改 vq_vae.py
                yield sample['field_data'], sample['hd_locs'], sample['mask']
            except StopIteration:
                active_dealers.pop(idx)
                while pending_projects and len(active_dealers) < self.nums_loading_projects:
                    p_dir, p_name = pending_projects.pop(0)
                    new_dealer = self._create_dealer(p_dir, p_name)
                    if new_dealer: active_dealers.append(new_dealer)

if __name__ == "__main__":
    import time
    print("="*60)
    print("🚀 测试动态 DatasetVQVAE3D (热加载与 Train/Val 批分)...")
    
    PROJECT_DIR = r"D:\\YihengZhu\\jbgs-lma\\proj-phase1\\eclipse-models\\1c\\water_flooding"
    PROJECT_NAME = "BYEPD93"
    
    # 模拟手头有 6 个宏观油藏项目文件
    p_dirs = [PROJECT_DIR] * 6
    p_names = [PROJECT_NAME] * 6
    
    print("\n[1] 实例化 Train 和 Val 视图 (零开销)...")
    t0 = time.time()
    
    # 训练集：加载压力、含水饱和度、含油饱和度等 5 通道，牌桌并发 2 个项目
    train_dataset = DatasetVQVAE3D(
        p_dirs, p_names, field_names=["PRESSURE", "SWAT", "SGAS"], 
        mode="train", nums_loading_projects=2
    )
    
    # 验证集：关闭 shuffle
    val_dataset = DatasetVQVAE3D(
        p_dirs, p_names, field_names=["PRESSURE", "SWAT", "SGAS"], 
        mode="val", nums_loading_projects=2, shuffle=False
    )
    
    print(f"✅ 视图实例化耗时: {time.time() - t0:.4f} 秒")
    
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0)
    
    print("\n[2] 开始抽取 Train DataLoader (验证时间步打乱与 VQ-VAE 的 3 元素解包)...")
    t_start = time.time()
    for batch_idx, batch in enumerate(train_loader):
        x, hard_data_locs, mask = batch
        print(f"  📦 Train Batch {batch_idx + 1} | Field (C=5): {x.shape} | HD: {hard_data_locs.shape}")
        if batch_idx == 1: break
    print(f"✅ Train 抽取耗时: {time.time() - t_start:.3f} 秒")

    print("\n[3] 开始抽取 Val DataLoader...")
    t_start = time.time()
    for batch_idx, batch in enumerate(val_loader):
        x, hard_data_locs, mask = batch
        print(f"  📦 Val Batch {batch_idx + 1} | Field (C=5): {x.shape}")
        if batch_idx == 0: break
    print(f"✅ Val 抽取耗时: {time.time() - t_start:.3f} 秒")
    print("="*60)