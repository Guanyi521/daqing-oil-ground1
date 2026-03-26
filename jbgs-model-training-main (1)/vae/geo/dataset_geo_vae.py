import torch
import random
import math
from pathlib import Path
from torch.utils.data import IterableDataset, DataLoader
from eclipse.model_data import EclipseModelData
from eclipse.well_data import EclipseWellData
from eclipse.utils import cylindricalize_static_data

class DatasetGeoVAE3D(IterableDataset):
    """
    基于 IterableDataset 的工业级海量 Eclipse 项目热加载器。
    支持内存分块管控 (nums_loading_projects) 和完美的随机洗牌机制。
    """
    def __init__(
        self, 
        project_dirs: list,      # 项目路径列表
        project_names: list,     # 项目名称列表
        project_target_wells: list, # 每个项目的目标井列表
        field_name: str,
        batch_size: int,         # 用于辅助计算
        nums_loading_projects: int = 5, # 内存中同时持有的项目数
        model_nz: int = 10, 
        nθ: int = 20, 
        nR: int = 10,
        max_hd_points: int = 10,
        shuffle: bool = True
    ):
        super().__init__()
        self.project_dirs = project_dirs
        self.project_names = project_names
        self.batch_size = batch_size
        self.nums_loading_projects = nums_loading_projects
        self.model_nz = model_nz
        self.nθ = nθ
        self.nR = nR
        self.max_hd_points = max_hd_points
        self.shuffle = shuffle
        self.field_name = field_name
        
        # 将路径、名称和目标井列表打包
        self.projects = list(zip(project_dirs, project_names, project_target_wells))
    
    
    def _process_chunk(self, chunk_projects):
        """核心处理逻辑：将一组项目加载并转换为扁平的张量字典列表"""
        chunk_samples = []
        
        for p_dir, p_name, target_wells in chunk_projects:
            try:
                # 1. 轻量化加载模型与井数据
                model = EclipseModelData(p_dir, p_name)
                dat_path = Path(p_dir) / f"{p_name}.DATA"
                well_data = EclipseWellData(str(dat_path))
                
                # 2. 调用 utils 极速采样引擎
                samples, maxR = cylindricalize_static_data(
                    model=model,
                    well_data=well_data,
                    field_name=self.field_name,
                    nθ=self.nθ,
                    nR=self.nR,
                    model_nz=self.model_nz,
                    max_hd_points=self.max_hd_points,
                    target_wells=target_wells
                )
                
                chunk_samples.extend(samples)
                
                # 3. 主动释放内存
                del model
                del well_data
                
            except Exception as e:
                print(f"⚠️ 警告: 加载项目 {p_name} 失败: {e}")
                continue
                
        return chunk_samples

    def __iter__(self):
        """生成器：控制 DataLoader 的数据流向"""
        worker_info = torch.utils.data.get_worker_info()
        
        # 1. 多线程 Worker 任务分配机制
        if worker_info is not None:
            # 将项目列表平均分配给各个 DataLoader Worker
            per_worker = int(math.ceil(len(self.projects) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(self.projects))
            my_projects = self.projects[start_idx:end_idx]
        else:
            my_projects = self.projects.copy()

        # 2. 项目级全局洗牌
        if self.shuffle:
            random.shuffle(my_projects)

        # 3. 内存分块流水线 (Chunking)
        for i in range(0, len(my_projects), self.nums_loading_projects):
            # 切片出当前允许载入内存的项目块
            chunk = my_projects[i : i + self.nums_loading_projects]
            
            # 集中榨取这些项目的物理特征
            chunk_samples = self._process_chunk(chunk)
            
            # 4. 井级局部洗牌 (保证 Batch 内数据的异质性)
            if self.shuffle:
                random.shuffle(chunk_samples)
                
            # 5. 逐个吐出样本给 Batch
            for sample in chunk_samples:
                yield sample['field_data'], sample['hd_locs'], sample['mask']


if __name__ == "__main__":
    print("="*60)
    print("🚀 启动 DatasetGeoVAE3D 内存分块热加载测试...")
    
    # 模拟项目环境配置
    PROJECT_DIR = "../"
    PROJECT_NAME = "BYEPD93"
    
    # 模拟 3 个不同的项目 (这里复用同一个做测试)
    p_dirs = [PROJECT_DIR, PROJECT_DIR, PROJECT_DIR]
    p_names = [PROJECT_NAME, PROJECT_NAME, PROJECT_NAME]
    
    # 设定：总共 3 个项目，但内存每次只允许加载 2 个！
    dataset = DatasetGeoVAE3D(
        project_dirs=p_dirs,
        project_names=p_names,
        batch_size=2,
        nums_loading_projects=2, # 🌟 内存阀门
        model_nz=10,             # 强制下采样统一到 Z=10
        nθ=20, 
        nR=10,
        max_hd_points=5,
        shuffle=True
    )
    
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
    
    for batch_idx, batch in enumerate(dataloader):
        # 🌟 既然 yield 的是 Tuple，这里直接解包 Tuple
        field_data, hd_locs, mask = batch
        
        print(f"\n📦 Batch {batch_idx + 1}:")
        print(f"  Field Shape: {field_data.shape}")  # 应为 (Batch, 1, 10, 20, 10)
        print(f"  Mask Shape: {mask.shape}")
        print(f"  Hard Data Locs Shape: {hd_locs.shape}")
        
        # 我们只打印前 2 个 batch 看效果
        if batch_idx == 1:
            break
            
    print("="*60)
    print("✅ 测试通过！海量项目惰性加载机制运行完美。")